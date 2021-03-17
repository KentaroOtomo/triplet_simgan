import numpy as np
import cv2
import gaze
from scipy.spatial.transform import Rotation as R
from gaze import vector_to_pitchyaw

def preprocess_unityeyes_image(img, json_data):
    #for 
    #convert PIL to ndarray
    #img = img.to('cpu').detach().numpy().copy()
    #img = np.array(img)
    #print(img,)
    img = np.array(img, dtype=np.uint8)
    ow = 60
    oh = 36
    # Prepare to segment eye image
    ih, iw = img.shape[:2]
    #ih_2, iw_2 = ih/2.0, iw/2.0
    ih_2, iw_2 = ih, iw

    #heatmap_w = int(ow/2)
    heatmap_w = int(ow)
    #heatmap_h = int(oh/2)
    heatmap_h = int(oh)


    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def process_coords(coords_list):
        #print(type([eval(l) for l in coords_list]))
        coords = [eval(l) for l in coords_list]
        #print(coords)
        return np.array([(x, 36-y, z) for (x, y, z) in coords])
        #%return np.array([(x, y) for (x, y) in coords])
        #return np.array([(x, y, z) for (x, y, z) in coords])
        #print(coords_list[0])

    interior_landmarks = process_coords(json_data['interior_margin_2d'])
    #print("interior_landmarks = " + str(interior_landmarks))
    caruncle_landmarks = process_coords(json_data['caruncle_2d'])
    iris_landmarks = process_coords(json_data['iris_2d'])
    #print(iris_landmarks.shape)
    '''
    i = 0
    while i <= 14:
        landmark = interior_landmarks[i][0]
        next_landmark = interior_landmarks[i+1][0]
        if landmark <= next_landmark:
            right_corner = next_landmark
            left_corner = landmark
        elif landmark >= next_landmark:
            right_corner = landmark
            left_corner = next_landmark     
    print('right = ', right_corner)
    print('left = ', left_corner)
    '''
    left_corner = np.mean(caruncle_landmarks[:, :2], axis=0)
    #%left_corner = interior_landmarks[0, :2]
    right_corner = interior_landmarks[8, :2]
    #print('left_corner:', left_corner)
    #print('right_corner:', right_corner)
    eye_width = 1.5 * abs(left_corner - right_corner)
    eye_middle = np.mean([np.amin(interior_landmarks[:, :2], axis=0),
                          np.amax(interior_landmarks[:, :2], axis=0)], axis=0)

    # Normalize to eye width.
    #scale = ow/eye_width
    scale = 1
    translate = np.asmatrix(np.eye(3))
    translate[0, 2] = -eye_middle[0] * scale
    translate[1, 2] = -eye_middle[1] * scale

    rand_x = np.random.uniform(low=0, high=0)
    rand_y = np.random.uniform(low=0, high=0)
    recenter = np.asmatrix(np.eye(3))
    recenter[0, 2] = ow/2 + rand_x
    #recenter[0, 2] = ow + rand_x
    recenter[1, 2] = oh/2 + rand_y
    #recenter[1, 2] = oh + rand_y

    scale_mat = np.asmatrix(np.eye(3))
    scale_mat[0, 0] = scale
    scale_mat[1, 1] = scale

    angle = 0 #np.random.normal(0, 1) * 20 * np.pi/180
    rotation = R.from_rotvec([0, 0, angle]).as_matrix()

    transform = recenter * rotation * translate * scale_mat
    transform_inv = np.linalg.inv(transform)
    
    
    # Apply transforms
    #eye = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eye = cv2.warpAffine(img, transform[:2], (ow, oh))

    rand_blur = np.random.uniform(low=0, high=20)
    eye = cv2.GaussianBlur(eye, (5, 5), rand_blur)

    # Normalize eye image
    eye = cv2.equalizeHist(eye)
    eye = eye.astype(np.float32)
    eye = eye / 255.0
    
    # Gaze
    # Convert look vector to gaze direction in polar angles
    gaze = np.array(eval(json_data['eye_details']['look_vec']))[:3].reshape((1, 3))
    ###look_vec = np.array(eval(json_data['eye_details']['look_vec']))[:3].reshape((1, 3))
    ###look_vec = np.matmul(look_vec, rotation.T)

    ###gaze = vector_to_pitchyaw(-look_vec).flatten()
    gaze = gaze.astype(np.float32)
    #gaze = np.asarray(gaze)
    
    iris_center = np.mean(iris_landmarks[:, :2], axis=0)
    landmarks = np.concatenate([interior_landmarks[:, :2],  # 8
                                #iris_landmarks[:, :2],  # 8
                                iris_landmarks[::2, :2],  # 8
                                iris_center.reshape((1, 2)),
                                [[iw_2, ih_2]],  # Eyeball center
                                ])  # 18 in total
    #print(interior_landmarks)
    #print(iris_landmarks[:, :2])
    landmarks = np.asmatrix(np.pad(landmarks, ((0, 0), (0, 1)), 'constant', constant_values=1))
    landmarks = np.asarray(landmarks * transform[:2].T) * np.array([heatmap_w/ow, heatmap_h/oh])
    landmarks = landmarks.astype(np.float32)
    landmarks = np.asarray(landmarks)
    #print('landmarks = ' + str(landmarks))

    
    # Swap columns so that landmarks are in (y, x), not (x, y)
    # This is because the network outputs landmarks as (y, x) values.
    temp = np.zeros((34, 2), dtype=np.float32)
    #print(landmarks[:, 1].shape)
    temp[:, 0] = landmarks[:, 1]
    temp[:, 1] = landmarks[:, 0]
    landmarks = temp
    #print(landmarks)
    heatmaps = get_heatmaps(w=heatmap_w, h=heatmap_h, landmarks=landmarks)
    heatmaps = np.asarray(heatmaps)
    assert heatmaps.shape == (34, heatmap_h, heatmap_w)
    
    """
    return {
        #'img': eye,
        #'transform': np.asarray(transform),
        #'transform_inv': np.asarray(transform_inv),
        #'eye_middle': np.asarray(eye_middle),
        np.asarray(heatmaps),
        np.asarray(landmarks),
        np.asarray(gaze)
    }
    """
    return heatmaps, landmarks, gaze
    #return heatmaps, landmarks

def gaussian_2d(w, h, cx, cy, sigma=2.0):
    """Generate heatmap with single 2D gaussian."""
    xs, ys = np.meshgrid(
        np.linspace(0, w - 1, w, dtype=np.float32),
        np.linspace(0, h - 1, h, dtype=np.float32)
    )

    assert xs.shape == (h, w)
    alpha = -0.5 / (sigma ** 2)
    heatmap = np.exp(alpha * ((xs - cx) ** 2 + (ys - cy) ** 2))
    return heatmap


def get_heatmaps(w, h, landmarks):
    heatmaps = []
    for (y, x) in landmarks:
        heatmaps.append(gaussian_2d(w, h, cx=x, cy=y, sigma=2.0))
    #print(heatmaps)    
    return np.array(heatmaps)

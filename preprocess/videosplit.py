import cv2
import os
import pandas as pd
from PIL import Image

video_path = ''
save_dir = ''

def save_all_frames(video_path, dir_path, basename, ext='png'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    n = 1
#print(digit)
#print(frame)

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}.{}'.format(str(n), ext), frame)
            n += 1
        else:
            return

#print(frame)

save_all_frames(video_path, save_dir, '')



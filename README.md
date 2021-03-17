Triplet SimGAN(Image Generator+Landmark Detector)

Usage
1.Create a virtual environment
　virtualenv env_name -p python3
  pip3 install -r requirements.txt

2.Download dataset
  You can download real image dataset and fake image and json dataset from dl001/dataset.
  (※Since the original dataset and the MPII Gaze Dataset have different image sizes, please modify them accordingly.)
  If you want to create a dataset with UnityEyes, you can download it from here.(Only windows.)
  https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/
  After creating the data, you can use preprocess/unityeyes.py and convertjson.py to create the training data.
  
　When creating real image data for training from video data, first input the video into OpenFace.
　https://github.com/TadasBaltrusaitis/OpenFace/wiki
  The first step is to split the video into frames. (preprocess/videosplit.py)
  Next, you create a crop image from the .csv file output from OpenFace and the frame-split image.(preprocess/real_crop.py)
   
3.Customize the config file and run the training
　You will need to edit the config file if necessary.(ex.Change directories, adjust hyperparameters)
　Next, go to the simgan directory and run main.py to start the training.

4.Running the test code
　Test it after you finish your training.
　First, modify the contents of main.py according to comments.
  When you run main.py, it will run the test and output a csv file that contains the center coordinates of the landmarks and the two-dimensional gaze direction (rad).

If you have any questions, please comment on the Issue or contact us at the address below.
soloist.jackson@gmail.com
　
　   

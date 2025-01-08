# -*- coding: utf-8 -*-
"""
Please make sure to pip install xgboost=2.1.3 and scikit-learn=1.5.2 which are compatible with py-feat.
"""
from feat import Detector
import pickle


detector = Detector()
detector

# path
path_video = " "
path_save = " "

# You can change skip_frames parameter
test_video_path = path_video
video_prediction = detector.detect_video(
    test_video_path, data_type="video", skip_frames=120, face_detection_threshold=0.95
)

# save the result
with open(path_save, 'wb') as f:
    pickle.dump(video_prediction, f)

# load the saved result
with open(path_save, 'rb') as f:
    video_prediction = pickle.load(f)

# Check how the data looks like
print(prediction_load.head())
print(prediction_load.shape)

# Now you can access specific groups of data

# face boxes: a rectangular bounding box of the face and includes a confidence score for each detected face.
prediction_load.faceboxes

# AU: Action Unit
prediction_load.aus

# emotions
prediction_load.emotions

# Head pose detection: rotations from a head on view can be described in terms of rotation around the x, y, and z planes and are referred to as pitch, roll, and yaw respectively
prediction_load.poses

# identities
prediction_load.identities


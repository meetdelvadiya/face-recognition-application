# Real-Time Face Recognition using Dlib and OpenCV  

## Overview  
This project implements a real-time face recognition application using Dlib and OpenCV. It captures video from the webcam, compares captured faces with a provided reference image, and displays whether the detected face matches the reference.  

## Features  
- Real-time face detection and recognition.  
- Uses Dlib for facial landmark detection and face encoding.  
- Utilizes OpenCV for webcam access and image processing.  

## Requirements  
- Python 3.x  
- Required libraries:  
  - OpenCV  
  - Dlib  
  - NumPy  

- You need to download these two files:  
  - [dlib_face_recognition_resnet_model_v1.dat](https://github.com/ageitgey/face_recognition_models/blob/master/face_recognition_models/models/dlib_face_recognition_resnet_model_v1.dat)
  - [shape_predictor_68_face_landmarks.dat](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)

You can install the required libraries using pip:  

```bash  
pip install opencv-python dlib numpy  

import dlib  
import numpy as np  
import cv2  
import os  

# Loading models with error handling  
def load_model(file_name):  
    if not os.path.isfile(file_name):  
        raise FileNotFoundError(f"Model file not found at: {file_name}")  
    return dlib.shape_predictor(file_name) if 'shape_predictor' in file_name else dlib.face_recognition_model_v1(file_name)  

def getFace(img):  
    face_detector = dlib.get_frontal_face_detector()  
    faces = face_detector(img, 1)  
    return faces[0] if faces else None  

def encodeFace(image, pose_predictor, face_encoder):  
    face_location = getFace(image)  
    if face_location is None:  
        return None  
    face_landmarks = pose_predictor(image, face_location)  
    face = dlib.get_face_chip(image, face_landmarks)  
    encodings = np.array(face_encoder.compute_face_descriptor(face))  
    return encodings  

def getSimilarity(image1, image2, pose_predictor, face_encoder):  
    face1_embeddings = encodeFace(image1, pose_predictor, face_encoder)  
    face2_embeddings = encodeFace(image2, pose_predictor, face_encoder)  
    if face1_embeddings is None or face2_embeddings is None:  
        return None  
    return np.linalg.norm(face1_embeddings - face2_embeddings)  

# Load models with error handling  
try:  
    pose_predictor = load_model('shape_predictor_68_face_landmarks.dat')  
    face_encoder = load_model('dlib_face_recognition_resnet_model_v1.dat')  
except FileNotFoundError as e:  
    print(e)  
    exit(1)  # Exit if the model files are not available  

# Load the reference image  
reference_image = cv2.imread('your_photo.jpg')  # Change to your actual reference image  
if reference_image is None:  
    print("Error: Reference image not found.")  
    exit(1)  # Exit if the reference image is not found  

# Optional: resize reference image for consistent processing speed  
reference_image = cv2.resize(reference_image, (320, 240))  

# Initialize webcam  
cap = cv2.VideoCapture(0)  
if not cap.isOpened():  
    print("Error: Could not open webcam.")  
    exit(1)  # Exit if the webcam cannot be opened  

while True:  
    # Capture frame-by-frame  
    ret, frame = cap.read()  
    if not ret:  
        print("Warning: Failed to capture frame.")  
        break  

    # Optional: resize frame for consistent processing speed  
    frame_resized = cv2.resize(frame, (320, 240))  

    # Calculate similarity  
    distance = getSimilarity(frame_resized, reference_image, pose_predictor, face_encoder)  

    # Check if similarity is valid  
    if distance is not None:  
        if distance < 0.6:  
            cv2.putText(frame, "Faces Match: Same Person", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,   
                        1, (0, 255, 0), 2, cv2.LINE_AA)  
        else:  
            cv2.putText(frame, "Faces Do Not Match: Different People", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,   
                        1, (0, 0, 255), 2, cv2.LINE_AA)  

    # Display the resulting frame  
    cv2.imshow('Webcam Feed', frame)  

    # Break the loop on 'q' key press  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  

# When everything done, release the capture  
cap.release()  
cv2.destroyAllWindows()  
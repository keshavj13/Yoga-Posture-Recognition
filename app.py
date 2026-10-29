
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Initialize mediapipe pose class
mp_pose = mp.solutions.pose

# Setting up the Pose function
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 


def custom_math_enhancement(image, gamma=0.4):
    '''
    Custom implementation of Power-Law (Gamma) Transformation.
    Demonstrates spatial domain matrix operations from scratch.
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Normalize the brightness matrix to float values between [0.0, 1.0]
    v_normalized = v / 255.0
    
    # Apply the mathematical power-law transformation matrix operation
    v_math_transformed = np.power(v_normalized, gamma)
    
    # Rescale the matrix back to [0, 255] and cast to 8-bit integers
    v_enhanced = np.uint8(v_math_transformed * 255.0)
    
    # Reconstruct the image from the manipulated channels
    hsv_enhanced = cv2.merge((h, s, v_enhanced))
    return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)


def detectPose(image, pose, display=True):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))
    
    return output_image, landmarks


def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    if angle < 0:
        angle += 360
    
    return angle


def classifyPose(landmarks, output_image, display=False):
    label = 'Unknown Pose'
    color = (0, 0, 255)
    
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    if (165 < left_knee_angle < 195) and (165 < right_knee_angle < 195) \
        and (130 < left_elbow_angle < 180) and (175 < right_elbow_angle < 220) \
        and (100 < left_shoulder_angle < 200) and (50 < right_shoulder_angle < 130):
        label = 'T Pose'
    
    if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:
                    label = 'Warrior II Pose' 
                        
    if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
        if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 25 and right_knee_angle < 45:
            label = 'Tree Pose'

    if label != 'Unknown Pose':
        color = (0, 255, 0)  
    
    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    # Always return the tuple of (output_image, label)
    return output_image, label


def webcam_feed():
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 1380)
    camera_video.set(4, 960)

    while camera_video.isOpened():
        ok, frame = camera_video.read()

        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

        # 1. Keep raw frame for the "Before" view
        raw_frame = frame.copy()
        cv2.putText(raw_frame, "Before: Raw Matrix", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        # 2. Process image using YOUR custom mathematical function
        enhanced_frame = custom_math_enhancement(frame, gamma=0.4)

        # 3. Detect and classify on the mathematically enhanced frame
        enhanced_frame, landmarks = detectPose(enhanced_frame, pose, display=False)

        if landmarks:
            enhanced_frame, _ = classifyPose(landmarks, enhanced_frame, display=False)
            
        cv2.putText(enhanced_frame, "After: Custom Power-Law Transform", (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # 4. Stack frames horizontally
        combined_frame = np.hstack((raw_frame, enhanced_frame))

        # Convert the combined frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', combined_frame)
        yield_frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + yield_frame + b'\r\n')

    camera_video.release()
    cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/yoga_try')
def yoga_try():
    return render_template('yoga_try.html')

@app.route('/video_feed1')
def video_feed1():
    return Response(webcam_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
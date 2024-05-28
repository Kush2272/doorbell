import cv2
import numpy as np
import face_recognition
from datetime import datetime

# Load image
kid_image = face_recognition.load_image_file('test1.jpeg')
kid_encoding = face_recognition.face_encodings(kid_image)[0]

# Initialize counters
match_counter = 0
unrecognized_counter = 0
recognized_counter = 0

# Generate reports
def update_counters(match, match_percentage):
    global match_counter, unrecognized_counter, recognized_counter
    if match:
        match_counter += 1
        recognized_counter += 1
        # Log the recognition information to a file
        with open('recognition_report.txt', 'a') as report_file:
            report_file.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Face recognized with {match_percentage:.2f}% match\n')
    else:
        unrecognized_counter += 1
        # Log the unrecognized face information to a file
        with open('recognition_report.txt', 'a') as report_file:
            report_file.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} - Unrecognized face detected with {match_percentage:.2f}% match\n')

# Size of box and tracking
def calculate_distance(face_location):
    average_face_height_cm = 21.0
    focal_length_mm = 20
    focal_length_pixel = 530
    # Calculate distance using the formula: distance = (focal_length * real_height) / (face_height_in_pixels * sensor_height)
    distance_cm = (focal_length_mm * average_face_height_cm) / (face_location[2] * 0.0264583333)  # Sensor height for typical webcam
    return distance_cm

# Open video camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert frame to RGB (required by face_recognition library)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    # Check if any face is detected in the frame
    if len(face_encodings) > 0:
        # Compare with the face
        match = face_recognition.compare_faces([kid_encoding], face_encodings[0], tolerance=0.48)[0]  # Adjust tolerance here
        
        # Update counters and generate report
        face_distance = face_recognition.face_distance([kid_encoding], face_encodings[0])
        match_percentage = (1 - face_distance[0]) * 100  # Corrected calculation
        update_counters(match, match_percentage)
        
        # Set label based on the match result
        if match:
            label = f'Image Matched ({match_percentage:.2f}% match)'
            color = (0, 255, 0)  # Green color
        else:
            label = f'Image Not Matched ({match_percentage:.2f}% match)'
            color = (0, 0, 255)  # Red color
        
        # Draw bounding box and label for the first detected face
        (top, right, bottom, left) = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Calculate and display distance
        distance = calculate_distance(face_locations[0])
        cv2.putText(frame, f'Distance: {distance:.2f} cm', (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()

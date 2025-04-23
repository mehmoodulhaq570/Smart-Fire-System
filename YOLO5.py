import cv2
import math
import time
import os
import datetime
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from ultralytics import YOLO

# ======================= GPIO & Servo Setup ========================
GPIO.setmode(GPIO.BCM)

SERVO_PIN_1 = 17  # Pan (left-right)
SERVO_PIN_2 = 27  # Tilt (up-down)

GPIO.setup(SERVO_PIN_1, GPIO.OUT)
GPIO.setup(SERVO_PIN_2, GPIO.OUT)

pwm1 = GPIO.PWM(SERVO_PIN_1, 50)
pwm2 = GPIO.PWM(SERVO_PIN_2, 50)
pwm1.start(0)
pwm2.start(0)

# ======================= YOLOv8 & Camera Setup =======================
model = YOLO("best.pt")  # Replace with your YOLOv8 model path

picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 1280)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

frame_width = 1280
frame_height = 1280
origin_x, origin_y = frame_width / 2, frame_height / 2

KNOWN_WIDTH = 1.0
FOCAL_LENGTH = 0.36
CAMERA_HORIZONTAL_FOV = 62  # degrees

# ======================= Helper Functions =======================

def set_angle(angle, pwm):
    duty = 2.5 + (angle / 18)
    duty = max(0.0, min(duty, 12.5))
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    pwm.ChangeDutyCycle(0)
    print(f"Servo moved to {angle} degrees with duty cycle {duty:.2f}%")

def calculate_angle_from_center(x):
    dx = x - origin_x
    angle_offset = (dx / frame_width) * CAMERA_HORIZONTAL_FOV
    servo_angle = 90 + angle_offset
    return int(max(0, min(180, servo_angle)))

def calculate_vertical_angle_from_height(bbox_height, min_height=50, max_height=500, min_angle=0, max_angle=120):
    """
    This function estimates the vertical servo angle based on the height of the detected object
    within the bounding box. The height will be inversely proportional to the distance.
    """

# Clamp bounding box height to within the min and max values
    bbox_height = max(min_height, min(bbox_height, max_height))
    
    # Scale the height to a corresponding vertical angle
    ratio = (bbox_height - min_height) / (max_height - min_height)
    angle = max_angle - ratio * (max_angle - min_angle)  # Inverted for a natural tilt movement
    
    # Ensure the angle is within the allowable range
    return int(max(min_angle, min(max_angle, angle)))

def capture_and_save_image(picam2, frame, h_angle, v_angle, distance, folder="captured_images"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"frame_{timestamp}.jpg")

    overlay = frame.copy()
    cv2.putText(overlay, f"Pan (Horizontal): {h_angle}°", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(overlay, f"Tilt (Vertical): {v_angle}°", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(overlay, f"Distance: {distance:.2f} cm", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    cv2.imwrite(filename, overlay)
    print(f"Image saved at {filename}")

# ======================= Main Loop =======================
try:
    while True:
        frame = picam2.capture_array()
        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()

                if conf > 0.5:
                    bbox_center_x = (x1 + x2) / 2
                    bbox_center_y = (y1 + y2) / 2
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1  # Get bounding box height
                    
                    # You can still calculate the distance using the width for horizontal movement
                    estimated_distance = (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width
                    real_distance = (estimated_distance * 2300 * 1.5) + 10
                    
                    # Calculate horizontal servo angle
                    horizontal_angle = calculate_angle_from_center(bbox_center_x)

                    # Use the new vertical angle calculation based on the bounding box height
                    vertical_angle = calculate_vertical_angle_from_height(bbox_height)

                    # Move servos
                    set_angle(180 - horizontal_angle, pwm1)  # Pan (inverted)
                    set_angle(vertical_angle, pwm2)  # Tilt

                    # Draw bounding box and overlay information
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (int(bbox_center_x), int(bbox_center_y)), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"Pan (Horizontal): {180 - horizontal_angle}°", (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(frame, f"Tilt (Vertical): {vertical_angle}°", (x1, y1 - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(frame, f"Distance: {real_distance:.2f} cm", (x1, y1 - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                    # Debug console output
                    print(f"Detection Center: ({bbox_center_x:.2f}, {bbox_center_y:.2f})")
                    print(f"Pan Angle (Horizontal): {180 - horizontal_angle}°")
                    print(f"Tilt Angle (Vertical): {vertical_angle}°")
                    print(f"Estimated Distance: {real_distance:.2f} cm")

                    # Save annotated image
                    capture_and_save_image(picam2, frame, 180 - horizontal_angle, vertical_angle, real_distance)

        cv2.imshow("YOLOv8 Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    pwm1.stop()
    pwm2.stop()
    GPIO.cleanup()
    print("Resources released.")

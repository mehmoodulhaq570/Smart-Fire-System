import cv2
import time
import os
import datetime
import threading
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np

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
model = YOLO("best_new.onnx")  # Replace with your YOLOv8 model path

picam2 = Picamera2()
picam2.preview_configuration.main.size = (480, 480)  # Low resolution for faster FPS
# picam2.create_video_configuration(main={'size':(640,480)}) for recording video purpoe
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

frame_width = 640
frame_height = 640
origin_x, origin_y = frame_width / 2, frame_height / 2

KNOWN_WIDTH = 1.0
FOCAL_LENGTH = 0.36
CAMERA_HORIZONTAL_FOV = 62  # degrees

# ======================= Helper Functions =======================

def set_angle(angle, pwm):
    """Set the servo angle and move the servo motor."""
    duty = 2.5 + (angle / 18)
    duty = max(0.0, min(duty, 12.5))
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)  # Allow servo to move
    pwm.ChangeDutyCycle(0)
    print(f"Servo moved to {angle} degrees with duty cycle {duty:.2f}%")

def calculate_angle_from_center(x):
    """Calculate horizontal pan angle based on object's x position."""
    dx = x - origin_x
    angle_offset = (dx / frame_width) * CAMERA_HORIZONTAL_FOV
    servo_angle = 90 + angle_offset
    return int(max(0, min(180, servo_angle)))

def calculate_vertical_angle(distance_cm, min_distance=30, max_distance=300, min_angle=30, max_angle=90):
    """Calculate vertical tilt angle based on object distance."""
    distance_cm = max(min_distance, min(distance_cm, max_distance))  # Clamp distance
    ratio = (distance_cm - min_distance) / (max_distance - min_distance)
    angle = max_angle - ratio * (max_angle - min_angle)  # Inverted for natural tilt
    return int(max(min_angle, min(max_angle, angle)))

def capture_and_save_image(picam2, frame, h_angle, v_angle, distance, folder="captured_images"):
    """Capture and save image with overlayed angle and distance information."""
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

# ======================= Frame Capture Thread ========================
frame = None
frame_lock = threading.Lock()

def capture_frame():
    """Capture a frame from the camera."""
    global frame
    while True:
        with frame_lock:
            frame = picam2.capture_array()

# ======================= Main Loop =======================
try:
    capture_thread = threading.Thread(target=capture_frame, daemon=True)
    capture_thread.start()

    while True:
        start_time = time.time()

        # Ensure thread-safe frame access
        with frame_lock:
            if frame is None:
                continue  # Skip if frame is not ready yet

        # YOLOv8 inference (running detection model)
        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()

                if conf > 0.5:
                    bbox_center_x = (x1 + x2) / 2
                    bbox_center_y = (y1 + y2) / 2

                    bbox_width = x2 - x1
                    estimated_distance = (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width
                    real_distance = (estimated_distance * 2300 * 1.5) + 10

                    # Calculate angles
                    horizontal_angle = calculate_angle_from_center(bbox_center_x)
                    vertical_angle = calculate_vertical_angle(real_distance)

                    # Move servos based on calculated angles
                    set_angle(180 - horizontal_angle, pwm1)  # Pan (horizontal)
                    set_angle(vertical_angle, pwm2)          # Tilt (vertical)

                    # Draw bounding box and overlay info
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
#                     capture_and_save_image(picam2, frame, 180 - horizontal_angle, vertical_angle, real_distance)

        # Display FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
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

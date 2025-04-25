import os
import time
import cv2
import datetime
import threading
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from ultralytics import YOLO

# ==================== Configuration =====================
SERVO_PIN_PAN = 17
SERVO_PIN_TILT = 27
CAMERA_RESOLUTION = (480, 480)
CAMERA_HORIZONTAL_FOV = 62  # degrees
FOCAL_LENGTH = 0.36
KNOWN_WIDTH = 1.0  # in cm (adjust for your target object)
FRAME_DISPLAY_SIZE = (640, 640)

# ==================== GPIO Setup ========================
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN_PAN, GPIO.OUT)
GPIO.setup(SERVO_PIN_TILT, GPIO.OUT)
pwm_pan = GPIO.PWM(SERVO_PIN_PAN, 50)
pwm_tilt = GPIO.PWM(SERVO_PIN_TILT, 50)
pwm_pan.start(0)
pwm_tilt.start(0)

# ==================== Camera & YOLO Init =================
picam2 = Picamera2()
picam2.preview_configuration.main.size = CAMERA_RESOLUTION
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

model = YOLO("best_new.onnx")  # Replace with your model path

origin_x, origin_y = FRAME_DISPLAY_SIZE[0] / 2, FRAME_DISPLAY_SIZE[1] / 2

# ==================== Frame Management ===================
frame = None
frame_lock = threading.Lock()

def frame_capture_thread():
    global frame
    while True:
        new_frame = picam2.capture_array()
        with frame_lock:
            frame = cv2.resize(new_frame, FRAME_DISPLAY_SIZE)

threading.Thread(target=frame_capture_thread, daemon=True).start()

# ==================== Helper Functions ===================

def set_servo_angle(angle, pwm):
    duty = 2.5 + (angle / 18)
    duty = max(2.5, min(duty, 12.5))
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.1)
    pwm.ChangeDutyCycle(0)

def calculate_pan_angle(x):
    dx = x - origin_x
    angle_offset = (dx / FRAME_DISPLAY_SIZE[0]) * CAMERA_HORIZONTAL_FOV
    return int(max(0, min(180, 90 + angle_offset)))

def calculate_tilt_angle(distance, min_d=30, max_d=300, min_a=30, max_a=120):
    distance = max(min_d, min(distance, max_d))
    ratio = (distance - min_d) / (max_d - min_d)
    angle = max_a - ratio * (max_a - min_a)
    return int(angle)

def estimate_distance(bbox_width):
    return ((KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width) * 2300 * 1.5 + 10

# ==================== Main Loop ==========================
try:
    while True:
        start_time = time.time()

        with frame_lock:
            if frame is None:
                continue
            current_frame = frame.copy()

        results = model(current_frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                if conf < 0.5:
                    continue

                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                bbox_width = x2 - x1
                distance = estimate_distance(bbox_width)

                pan_angle = calculate_pan_angle(cx)
                tilt_angle = calculate_tilt_angle(distance)

                set_servo_angle(180 - pan_angle, pwm_pan)
                set_servo_angle(tilt_angle, pwm_tilt)

                # Drawing and annotation
                cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(current_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.putText(current_frame, f"Pan: {180 - pan_angle}°", (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(current_frame, f"Tilt: {tilt_angle}°", (x1, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(current_frame, f"Dist: {distance:.1f} cm", (x1, y1 - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(current_frame, f"FPS: {fps:.2f}", (10, FRAME_DISPLAY_SIZE[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Tracking", current_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    pwm_pan.stop()
    pwm_tilt.stop()
    GPIO.cleanup()
    print("Cleaned up GPIO and camera.")

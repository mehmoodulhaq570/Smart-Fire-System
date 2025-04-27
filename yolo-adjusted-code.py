# ============Conditions============
# This code is for educational purposes only. The author is not responsible for any damage or injury caused by the use of this code.

# 1 - First detect fire → set pan and tilt angles (based on center of the fire)
# 2 -  After setting angles → wait 3 seconds (fixed nozzle position)
# 3 - Then turn on water motor + buzzer (after 3 sec)
# 4 - Nozzle must remain stable even if the bounding box size or position slightly changes after detection.

import os
import cv2
import time
import datetime
import threading
import queue
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from ultralytics import YOLO

# ============ Configuration =============
SERVO_PIN_PAN = 17
SERVO_PIN_TILT = 27
MOTOR_ENA = 18  # Pin for PWM (Speed control)
MOTOR_IN1 = 23  # Direction control 1
MOTOR_IN2 = 24  # Direction control 2
BUZZER_PIN = 6  # Pin for buzzer
FRAME_SIZE = (480, 480)
DISPLAY_SIZE = (640, 640)
FPS_LIMIT = 10
FOCAL_LENGTH = 0.36
KNOWN_WIDTH = 1.0
CAMERA_HORIZONTAL_FOV = 62
CAMERA_VERTICAL_FOV = 60  # degrees

# ============ GPIO Setup =============
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN_PAN, GPIO.OUT)
GPIO.setup(SERVO_PIN_TILT, GPIO.OUT)
GPIO.setup(MOTOR_ENA, GPIO.OUT)
GPIO.setup(MOTOR_IN1, GPIO.OUT)
GPIO.setup(MOTOR_IN2, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

pwm_pan = GPIO.PWM(SERVO_PIN_PAN, 50)
pwm_tilt = GPIO.PWM(SERVO_PIN_TILT, 50)
pwm_motor = GPIO.PWM(MOTOR_ENA, 1000)
pwm_pan.start(0)
pwm_tilt.start(0)
pwm_motor.start(0)

# ============ Camera Setup =============
picam2 = Picamera2()
picam2.preview_configuration.main.size = FRAME_SIZE
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# ============ YOLOv8 Load =============
model = YOLO("best_new.onnx")

# ============ Globals =============
frame_queue = queue.Queue(maxsize=2)
origin_x, origin_y = DISPLAY_SIZE[0] / 2, DISPLAY_SIZE[1] / 2
running = True
fire_detected = False
fire_handling = False
last_detection_time = None
angles_set = False

# ============ Helper Functions =============
def set_servo_angle(angle, pwm):
    duty = 2.5 + (angle / 18)
    pwm.ChangeDutyCycle(max(2.5, min(duty, 12.5)))
    time.sleep(0.05)
    pwm.ChangeDutyCycle(0)

def calculate_pan_angle(x):
    dx = x - origin_x
    offset = (dx / DISPLAY_SIZE[0]) * CAMERA_HORIZONTAL_FOV
    return int(np.clip(90 + offset, 0, 180))

def calculate_tilt_angle(y, FRAME_DISPLAY_SIZE=FRAME_SIZE, CAMERA_VERTICAL_FOV=CAMERA_VERTICAL_FOV):
    origin_y = FRAME_DISPLAY_SIZE[1] // 2
    dy = y - origin_y
    angle_offset = (dy / FRAME_DISPLAY_SIZE[1]) * CAMERA_VERTICAL_FOV
    angle = 90 + angle_offset
    return int(max(0, min(180, angle)))

def estimate_distance(bbox_width):
    return ((KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width) * 2300 * 1.5 + 10

def set_motor_speed(distance):
    speed = max(0, min(100, (1500 / distance) - 10))
    pwm_motor.ChangeDutyCycle(speed)

def activate_motor_and_buzzer():
    GPIO.output(MOTOR_IN1, GPIO.HIGH)
    GPIO.output(MOTOR_IN2, GPIO.LOW)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)

def deactivate_motor_and_buzzer():
    pwm_motor.ChangeDutyCycle(0)
    GPIO.output(MOTOR_IN1, GPIO.LOW)
    GPIO.output(MOTOR_IN2, GPIO.LOW)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

# ============ Threads =============
def capture_thread():
    while running:
        frame = picam2.capture_array()
        if not frame_queue.full():
            resized = cv2.resize(frame, DISPLAY_SIZE)
            frame_queue.put(resized)

def processing_thread():
    global running, fire_detected, fire_handling, last_detection_time, angles_set
    last_pan_angle = 90
    last_tilt_angle = 90

    while running:
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()
        start_time = time.time()
        results = model(frame)

        current_fire_detected = False

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                if conf < 0.5:
                    continue

                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                bbox_width = x2 - x1

                if not angles_set:
                    pan_angle = calculate_pan_angle(cx)
                    tilt_angle = calculate_tilt_angle(cy)
                    last_pan_angle = pan_angle
                    last_tilt_angle = tilt_angle

                    # Set the servo angles for pan and tilt
                    set_servo_angle(180 - pan_angle, pwm_pan)
                    set_servo_angle(tilt_angle, pwm_tilt)

                    last_detection_time = time.time()
                    angles_set = True
                    print("Fire detected! Angles set. Waiting to activate motor and buzzer...")

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                current_fire_detected = True

        if current_fire_detected:
            # Only activate motor and buzzer after 3 seconds of fire detection
            if angles_set and (time.time() - last_detection_time >= 3):
                # Estimate distance and set motor speed
                distance = estimate_distance(bbox_width)
                set_motor_speed(distance)  # Adjust motor speed based on distance
                activate_motor_and_buzzer()
                print("Motor and buzzer activated!")
        else:
            if fire_detected:
                print("No fire detected anymore. Resetting...")
            angles_set = False
            deactivate_motor_and_buzzer()

        fire_detected = current_fire_detected

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, DISPLAY_SIZE[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOV11 Fire Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

        elapsed = time.time() - start_time
        sleep_time = max(0, (1.0 / FPS_LIMIT) - elapsed)
        time.sleep(sleep_time)

# ============ Start Threads =============
threading.Thread(target=capture_thread, daemon=True).start()
threading.Thread(target=processing_thread, daemon=True).start()

# ============ Main Wait Loop ============
try:
    while running:
        time.sleep(1)

except KeyboardInterrupt:
    print("User Interrupted!")

finally:
    running = False
    time.sleep(1)
    picam2.stop()
    pwm_pan.stop()
    pwm_tilt.stop()
    pwm_motor.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
    print("Clean exit.")

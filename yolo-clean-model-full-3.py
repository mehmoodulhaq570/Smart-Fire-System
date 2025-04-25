# This code has the previous integrations and motor code is also adjsuted it

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
FRAME_SIZE = (480, 480)
DISPLAY_SIZE = (640, 640)
FPS_LIMIT = 10
FOCAL_LENGTH = 0.36
KNOWN_WIDTH = 1.0
CAMERA_HORIZONTAL_FOV = 62
CAMERA_VERTICAL_FOV = 60  # degrees (adjust based on camera)

# ============ GPIO Setup =============
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN_PAN, GPIO.OUT)
GPIO.setup(SERVO_PIN_TILT, GPIO.OUT)
GPIO.setup(MOTOR_ENA, GPIO.OUT)
GPIO.setup(MOTOR_IN1, GPIO.OUT)
GPIO.setup(MOTOR_IN2, GPIO.OUT)

pwm_pan = GPIO.PWM(SERVO_PIN_PAN, 50)
pwm_tilt = GPIO.PWM(SERVO_PIN_TILT, 50)
pwm_motor = GPIO.PWM(MOTOR_ENA, 1000)  # 1 kHz frequency for motor PWM
pwm_pan.start(0)
pwm_tilt.start(0)
pwm_motor.start(0)  # Start with 0% duty cycle (motor off)

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

# Motor control function to adjust speed based on distance
def set_motor_speed(distance):
    # Adjust the motor speed based on distance (the closer, the faster)
    speed = max(0, min(100, (1500 / distance) - 10))  # You can tweak the equation based on your needs
    pwm_motor.ChangeDutyCycle(speed)

# ============ Threads =============
def capture_thread():
    while running:
        frame = picam2.capture_array()
        if not frame_queue.full():
            resized = cv2.resize(frame, DISPLAY_SIZE)
            frame_queue.put(resized)

def processing_thread():
    global running
    while running:
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()
        start_time = time.time()
        results = model(frame)

        fire_detected = False  # Flag to check if fire is detected in the frame

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
                tilt_angle = calculate_tilt_angle(cy)

                # Move servo motors
                set_servo_angle(180 - pan_angle, pwm_pan)
                set_servo_angle(tilt_angle, pwm_tilt)

                # Adjust motor speed based on the estimated distance to the fire
                set_motor_speed(distance)

                # Annotate frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Pan: {180 - pan_angle}°", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(frame, f"Tilt: {tilt_angle}°", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(frame, f"Dist: {distance:.1f} cm", (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                fire_detected = True  # Set flag to True if fire is detected

        # If no fire is detected, turn off the motor
        if not fire_detected:
            pwm_motor.ChangeDutyCycle(0)
            print("No fire detected. Motor is OFF.")

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, DISPLAY_SIZE[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO Fire Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

        # Control frame rate
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
    pwm_motor.stop()  # Stop motor PWM
    GPIO.cleanup()
    cv2.destroyAllWindows()
    print("Clean exit.")

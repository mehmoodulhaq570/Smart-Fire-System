# Detect fire using YOLOv11 model.
# Monitor gas leak (MQ5) and flame sensor simultaneously.
# Show real-time status on LCD.
# Activate motor + buzzer if any (gas leak / flame / camera fire) is detected.
# The camera center and the servo center (rotation point) are vertically offset by 12.7 cm.

import os
import cv2
import time
import threading
import queue
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from ultralytics import YOLO
from RPLCD.i2c import CharLCD

# ============ Configuration ============= 
SERVO_PIN_PAN = 17
SERVO_PIN_TILT = 27
MOTOR_ENA = 18
MOTOR_IN1 = 23
MOTOR_IN2 = 24
BUZZER_PIN = 6
MQ5_PIN = 26
FLAME_PIN = 16

FRAME_SIZE = (480, 480)
DISPLAY_SIZE = (640, 640)
FPS_LIMIT = 10
FOCAL_LENGTH = 0.36
KNOWN_WIDTH = 1.0
CAMERA_HORIZONTAL_FOV = 62
CAMERA_VERTICAL_FOV = 60  # degrees

MOTOR_ACTIVATION_DURATION = 5  # seconds

# ============ GPIO Setup ============= 
GPIO.setmode(GPIO.BCM)
GPIO.setup([SERVO_PIN_PAN, SERVO_PIN_TILT, MOTOR_ENA, MOTOR_IN1, MOTOR_IN2, BUZZER_PIN], GPIO.OUT)
GPIO.setup([MQ5_PIN, FLAME_PIN], GPIO.IN)

pwm_pan = GPIO.PWM(SERVO_PIN_PAN, 50)
pwm_tilt = GPIO.PWM(SERVO_PIN_TILT, 50)
pwm_motor = GPIO.PWM(MOTOR_ENA, 1000)
pwm_pan.start(0)
pwm_tilt.start(0)
pwm_motor.start(0)

# LCD Setup
lcd = CharLCD('PCF8574', 0x27)  # Adjust I2C address if needed

# ============ Camera Setup ============ 
picam2 = Picamera2()
picam2.preview_configuration.main.size = FRAME_SIZE
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# ============ YOLOv11 Load ============ 
model = YOLO("best_new.onnx")

# ============ Globals ============ 
frame_queue = queue.Queue(maxsize=2)
running = True
fire_detected = False
gas_detected = False
flame_detected = False
activation_end_time = 0

# ============ Helper Functions ============ 
def set_servo_angle(angle, pwm):
    duty = 2.5 + (angle / 18)
    pwm.ChangeDutyCycle(max(2.5, min(duty, 12.5)))
    time.sleep(0.05)
    pwm.ChangeDutyCycle(0)

def calculate_pan_angle(x):
    dx = x - (DISPLAY_SIZE[0] / 2)
    offset = (dx / DISPLAY_SIZE[0]) * CAMERA_HORIZONTAL_FOV
    return int(np.clip(90 + offset, 0, 180))

def calculate_tilt_angle(y, distance):
    dy = y - (DISPLAY_SIZE[1] / 2)
    
    # Convert pixel displacement to vertical angle
    pixel_angle = (dy / DISPLAY_SIZE[1]) * CAMERA_VERTICAL_FOV
    
    # Adjust for 12.7 cm vertical offset
    vertical_offset_angle = np.degrees(np.arctan(12.7 / distance))
    
    total_angle = pixel_angle + vertical_offset_angle
    
    return int(np.clip(90 + total_angle, 0, 180))

def estimate_distance(bbox_width):
    if bbox_width == 0:
        return 9999
    return ((KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width) * 2300 * 1.5 + 10

def set_motor_speed(distance):
    speed = max(0, min(100, (1500 / distance) - 10))
    pwm_motor.ChangeDutyCycle(speed)

def activate_motor_and_buzzer():
    global activation_end_time
    GPIO.output(MOTOR_IN1, GPIO.HIGH)
    GPIO.output(MOTOR_IN2, GPIO.LOW)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    activation_end_time = time.time() + MOTOR_ACTIVATION_DURATION

def deactivate_motor_and_buzzer():
    if time.time() >= activation_end_time:
        pwm_motor.ChangeDutyCycle(0)
        GPIO.output(MOTOR_IN1, GPIO.LOW)
        GPIO.output(MOTOR_IN2, GPIO.LOW)
        GPIO.output(BUZZER_PIN, GPIO.LOW)

# ============ Threads ============ 
def capture_thread():
    while running:
        frame = picam2.capture_array()
        if not frame_queue.full():
            resized = cv2.resize(frame, DISPLAY_SIZE)
            frame_queue.put(resized)

def processing_thread():
    global fire_detected

    while running:
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()
        start_time = time.time()
        results = model(frame)

        fire_detected = False
        best_confidence = 0
        best_bbox = None

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                if conf < 0.5:
                    continue

                fire_detected = True
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                bbox_width = x2 - x1

                if conf > best_confidence:
                    best_confidence = conf
                    best_bbox = (cx, cy, bbox_width)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        if fire_detected and best_bbox:
            cx, cy, bbox_width = best_bbox
            pan_angle = calculate_pan_angle(cx)
            distance = estimate_distance(bbox_width)
            tilt_angle = calculate_tilt_angle(cy, distance)

            set_servo_angle(180 - pan_angle, pwm_pan)
            set_servo_angle(tilt_angle, pwm_tilt)
            set_motor_speed(distance)
            activate_motor_and_buzzer()

            print(f"🔥 Fire detected | Pan: {pan_angle}° | Tilt: {tilt_angle}° | Distance: {distance:.2f} cm")

        if not (fire_detected or gas_detected or flame_detected):
            deactivate_motor_and_buzzer()

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, DISPLAY_SIZE[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLO Fire Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(max(0, (1.0 / FPS_LIMIT) - (time.time() - start_time)))

def sensor_thread():
    global gas_detected, flame_detected

    while running:
        gas_status = GPIO.input(MQ5_PIN)
        flame_status = GPIO.input(FLAME_PIN)

        lcd.clear()
        if gas_status == GPIO.LOW:
            lcd.write_string("Gas: Detected")
            gas_detected = True
        else:
            lcd.write_string("Gas: Normal")
            gas_detected = False

        lcd.crlf()

        if flame_status == GPIO.LOW:
            lcd.write_string("Flame: Detected")
            flame_detected = True
        else:
            lcd.write_string("Flame: Safe")
            flame_detected = False

        if gas_detected or flame_detected:
            activate_motor_and_buzzer()
            print("⚠️ Sensor detected danger! Motor & buzzer activated.")

        if not (fire_detected or gas_detected or flame_detected):
            deactivate_motor_and_buzzer()

        time.sleep(1)

# ============ Start Threads ============ 
threads = [
    threading.Thread(target=capture_thread, daemon=True),
    threading.Thread(target=processing_thread, daemon=True),
    threading.Thread(target=sensor_thread, daemon=True)
]

for t in threads:
    t.start()

# ============ Main Program ============ 
try:
    while running:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🔴 Interrupted by user!")

finally:
    running = False
    time.sleep(2)  # Allow threads to finish
    picam2.stop()
    pwm_pan.stop()
    pwm_tilt.stop()
    pwm_motor.stop()
    GPIO.cleanup()
    lcd.clear()
    cv2.destroyAllWindows()
    print("✅ Clean exit. All resources released.")

import cv2
import time
import threading
import queue
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from ultralytics import YOLO

# ============ Configuration =============
SERVO_PIN_PAN = 17
SERVO_PIN_TILT = 27

FRAME_SIZE = (480, 480)
DISPLAY_SIZE = (640, 640)
FPS_LIMIT = 10
CAMERA_HORIZONTAL_FOV = 62  # degrees
CAMERA_VERTICAL_FOV = 60    # degrees

# ============ GPIO Setup =============
GPIO.setmode(GPIO.BCM)
GPIO.setup([SERVO_PIN_PAN, SERVO_PIN_TILT], GPIO.OUT)

pwm_pan = GPIO.PWM(SERVO_PIN_PAN, 50)
pwm_tilt = GPIO.PWM(SERVO_PIN_TILT, 50)
pwm_pan.start(0)
pwm_tilt.start(0)

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

def calculate_tilt_angle(y):
    dy = y - (DISPLAY_SIZE[1] / 2)
    offset = (dy / DISPLAY_SIZE[1]) * CAMERA_VERTICAL_FOV
    return int(np.clip(90 + offset, 0, 180))

# ============ Threads ============
def capture_thread():
    while running:
        frame = picam2.capture_array()
        if not frame_queue.full():
            resized = cv2.resize(frame, DISPLAY_SIZE)
            frame_queue.put(resized)

def processing_thread():
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

                if conf > best_confidence:
                    best_confidence = conf
                    best_bbox = (cx, cy)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        if fire_detected and best_bbox:
            cx, cy = best_bbox
            pan_angle = calculate_pan_angle(cx)
            tilt_angle = calculate_tilt_angle(cy)

            set_servo_angle(180 - pan_angle, pwm_pan)  # Adjust for mechanical orientation
            set_servo_angle(tilt_angle, pwm_tilt)

            print(f"🔥 Fire detected | Pan: {pan_angle}° | Tilt: {tilt_angle}°")

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, DISPLAY_SIZE[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Fire Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(max(0, (1.0 / FPS_LIMIT) - (time.time() - start_time)))

# ============ Start Threads ============
threads = [
    threading.Thread(target=capture_thread, daemon=True),
    threading.Thread(target=processing_thread, daemon=True)
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
    time.sleep(2)
    picam2.stop()
    pwm_pan.stop()
    pwm_tilt.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
    print("✅ Clean exit.")

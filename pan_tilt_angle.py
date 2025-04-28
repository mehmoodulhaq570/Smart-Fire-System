import cv2
import numpy as np
from ultralytics import YOLO
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# ===== Configuration =====
SERVO_PIN_PAN = 17
SERVO_PIN_TILT = 27

FRAME_SIZE = (480, 480)
DISPLAY_SIZE = (640, 640)

CAMERA_HORIZONTAL_FOV = 62  # degrees
CAMERA_VERTICAL_FOV = 60    # degrees
VERTICAL_OFFSET_CM = 12.7

# ===== GPIO Setup =====
GPIO.setmode(GPIO.BCM)
GPIO.setup([SERVO_PIN_PAN, SERVO_PIN_TILT], GPIO.OUT)

pwm_pan = GPIO.PWM(SERVO_PIN_PAN, 50)
pwm_tilt = GPIO.PWM(SERVO_PIN_TILT, 50)
pwm_pan.start(0)
pwm_tilt.start(0)

# ===== Camera Setup =====
picam2 = Picamera2()
picam2.preview_configuration.main.size = FRAME_SIZE
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# ===== Load YOLOv11 model =====
model = YOLO("best_new.onnx")

# ===== Helper Functions =====
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
    origin_y = DISPLAY_SIZE[1] // 2
    dy = y - origin_y

    angle_offset = (dy / DISPLAY_SIZE[1]) * CAMERA_VERTICAL_FOV
    angle_from_center = 90 + angle_offset

    # Calibration: adjust for 12.7 cm vertical offset
    if distance > 0:
        correction_angle = np.degrees(np.arctan(VERTICAL_OFFSET_CM / distance))
    else:
        correction_angle = 0

    corrected_angle = angle_from_center - correction_angle
    return int(np.clip(corrected_angle, 0, 180))

def estimate_distance(bbox_width):
    if bbox_width == 0:
        return 9999
    return ((1.0 * 0.36) / bbox_width) * 2300 * 1.5 + 10

# ===== Main Program =====
try:
    while True:
        frame = picam2.capture_array()
        resized_frame = cv2.resize(frame, DISPLAY_SIZE)
        results = model(resized_frame)

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

                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(resized_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        if fire_detected and best_bbox:
            cx, cy, bbox_width = best_bbox
            distance = estimate_distance(bbox_width)
            pan_angle = calculate_pan_angle(cx)
            tilt_angle = calculate_tilt_angle(cy, distance)

            set_servo_angle(180 - pan_angle, pwm_pan)  # 180 - pan to match servo mounting
            set_servo_angle(tilt_angle, pwm_tilt)

            print(f"🔥 Fire detected | Pan: {pan_angle}° | Tilt: {tilt_angle}° | Distance: {distance:.2f} cm")

        cv2.imshow("Fire Detection", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🔴 Interrupted by user!")

finally:
    picam2.stop()
    pwm_pan.stop()
    pwm_tilt.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
    print("✅ Clean exit. All resources released.")

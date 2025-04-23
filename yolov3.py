# This code with capture the image and save them in a directory
# No calucaltion of angle for second motor servo pin 27
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

SERVO_PIN_1 = 17
SERVO_PIN_2 = 27

GPIO.setup(SERVO_PIN_1, GPIO.OUT)
GPIO.setup(SERVO_PIN_2, GPIO.OUT)

pwm1 = GPIO.PWM(SERVO_PIN_1, 50)
pwm2 = GPIO.PWM(SERVO_PIN_2, 50)
pwm1.start(0)
pwm2.start(0)

# ======================= YOLOv8 & Camera Setup =======================
model = YOLO("best.pt")  # Replace with your actual YOLO model path

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
CAMERA_HORIZONTAL_FOV = 62  

# ======================= Helper Functions =======================

def set_angle(angle, pwm):
    duty = 2.5 + (angle / 18)
    duty = max(0.0, min(duty, 12.5))
    pwm.ChangeDutyCycle(duty)
    time.sleep(2.0)
    pwm.ChangeDutyCycle(0)
    print(f"Servo moved to {angle} degrees with duty cycle {duty:.2f}%")

def calculate_angle_from_center(x):
    dx = x - origin_x
    angle_offset = (dx / frame_width) * CAMERA_HORIZONTAL_FOV
    servo_angle = 90 + angle_offset
    return int(max(0, min(180, servo_angle)))

def capture_and_save_image(picam2, frame, angle, distance, folder="captured_images"):
    """Capture image and save with overlayed angle and distance."""
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"frame_{timestamp}.jpg")

    overlay = frame.copy()
    cv2.putText(overlay, f"Angle: {angle}°", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(overlay, f"Distance: {distance:.2f} cm", (10, 60),
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

                    servo_angle_1 = calculate_angle_from_center(bbox_center_x)
                    servo_angle_2 = calculate_angle_from_center(bbox_center_x)

                    # Reverse angle for SERVO_PIN_1
                    set_angle(180 - servo_angle_1, pwm1)
                    set_angle(servo_angle_2, pwm2)

                    bbox_width = x2 - x1
                    estimated_distance = (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width
                    real_distance = (estimated_distance * 2300 * 1.5) + 10

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (int(bbox_center_x), int(bbox_center_y)), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"Angle: {servo_angle_1}°", (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(frame, f"Distance: {real_distance:.2f} cm", (x1, y1 - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                    print(f"Detection Center: ({bbox_center_x:.2f}, {bbox_center_y:.2f})")
                    print(f"Servo Angle 1 (reversed): {180 - servo_angle_1}°")
                    print(f"Servo Angle 2: {servo_angle_2}°")
                    print(f"Estimated Distance: {real_distance:.2f} cm")

                    # Save the frame with overlayed info
                    capture_and_save_image(picam2, frame, servo_angle_1, real_distance)

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
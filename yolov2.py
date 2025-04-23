import cv2
import math
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2  # Import Picamera2 for Raspberry Pi camera
from ultralytics import YOLO

# ======================= GPIO & Servo Setup ========================
GPIO.setmode(GPIO.BCM)

# Define servo control pins
SERVO_PIN_1 = 17
SERVO_PIN_2 = 27

# Set up the servo pins
GPIO.setup(SERVO_PIN_1, GPIO.OUT)
GPIO.setup(SERVO_PIN_2, GPIO.OUT)

# Create PWM instances with 50Hz frequency
pwm1 = GPIO.PWM(SERVO_PIN_1, 50)
pwm2 = GPIO.PWM(SERVO_PIN_2, 50)
pwm1.start(0)
pwm2.start(0)

# ======================= YOLOv8 & Camera Setup =======================
model = YOLO("best.pt")

picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 1280)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Camera specs
frame_width = 1280
frame_height = 1280
origin_x, origin_y = frame_width / 2, frame_height / 2

# Known parameters (replace with actual values)
KNOWN_WIDTH = 1.0
FOCAL_LENGTH = 0.36

# Camera horizontal FoV in degrees (adjust if using different cam)
CAMERA_HORIZONTAL_FOV = 62  

# ======================= Helper Functions =======================

def set_angle(angle, pwm):
    """Convert angle (0–180) to duty cycle and send to servo"""
    duty = 2.5 + (angle / 18)
    duty = max(0.0, min(duty, 12.5))
    pwm.ChangeDutyCycle(duty)
    time.sleep(2.0)
    pwm.ChangeDutyCycle(0)
    print(f"Servo moved to {angle} degrees with duty cycle {duty:.2f}%")

def calculate_angle_from_center(x):
    """
    Calculate the angle from the image center to the object in the X-axis.
    Returns servo angle between 0 and 180.
    """
    dx = x - origin_x
    angle_offset = (dx / frame_width) * CAMERA_HORIZONTAL_FOV
    servo_angle = 90 + angle_offset
    return int(max(0, min(180, servo_angle)))

# ======================= Main Loop =======================
try:
    while True:
        frame = picam2.capture_array()

        # YOLO Inference
        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()

                if conf > 0.5:
                    bbox_center_x = (x1 + x2) / 2
                    bbox_center_y = (y1 + y2) / 2

                    # Calculate servo angles using new method
                    servo_angle_1 = calculate_angle_from_center(bbox_center_x)
                    servo_angle_2 = calculate_angle_from_center(bbox_center_x)

                    # Move servos
                    set_angle(servo_angle_1, pwm1)
                    set_angle(servo_angle_2, pwm2)

                    # Estimate distance
                    bbox_width = x2 - x1
                    estimated_distance = (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width
                    real_distance = (estimated_distance * 2300 * 1.5) + 10

                    # Draw box and annotations
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (int(bbox_center_x), int(bbox_center_y)), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"Angle: {servo_angle_1}°", (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(frame, f"Distance: {real_distance:.2f} cm", (x1, y1 - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                    # Debug
                    print(f"Detection Center: ({bbox_center_x:.2f}, {bbox_center_y:.2f})")
                    print(f"Servo Angle 1: {servo_angle_1}°")
                    print(f"Servo Angle 2: {servo_angle_2}°")
                    print(f"Estimated Distance: {real_distance:.2f} cm")

        # Show frame
        cv2.imshow("YOLOv8 Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    # Cleanup
    picam2.stop()
    cv2.destroyAllWindows()
    pwm1.stop()
    pwm2.stop()
    GPIO.cleanup()
    print("Resources released.")
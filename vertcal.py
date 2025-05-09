import cv2
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from ultralytics import YOLO

# ======================= GPIO & Servo Setup =======================
GPIO.setmode(GPIO.BCM)

SERVO_PIN = 27
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz frequency for servo motors
pwm.start(0)

# ======================= YOLOv8 & Camera Setup =======================
model = YOLO("best_new.onnx")  # Replace with your trained YOLOv8 model

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# ======================= Helper Functions =======================

def set_angle(angle):
    duty = 2.5 + (angle / 18)
    duty = max(0.0, min(duty, 12.5))
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    pwm.ChangeDutyCycle(0)
    print(f"Servo moved to {angle} degrees with duty cycle {duty:.2f}%")

def calculate_vertical_angle_from_height(bbox_height, min_height=30, max_height=300, min_angle=30, max_angle=120):
    """
    Estimate vertical angle for servo based on bounding box height.
    Larger bbox -> closer object -> servo tilts upward (higher angle).
    """
    # Clamp bounding box height within expected range
    bbox_height = max(min_height, min(bbox_height, max_height))
    ratio = (bbox_height - min_height) / (max_height - min_height)
    angle = max_angle - ratio * (max_angle - min_angle)
    return int(max(min_angle, min(max_angle, angle)))

# ======================= Main Loop =======================
try:
    last_angle = None

    while True:
        frame = picam2.capture_array()
        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()

                if conf > 0.5:
                    bbox_height = y2 - y1
                    vertical_angle = calculate_vertical_angle_from_height(bbox_height)

                    # Only move servo if angle has changed significantly
                    if last_angle is None or abs(vertical_angle - last_angle) > 2:
                        set_angle(vertical_angle)
                        last_angle = vertical_angle

                    # Draw detection info
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Tilt: {vertical_angle}°", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    print(f"Detection Height: {bbox_height}")
                    print(f"Tilt Angle (Vertical): {vertical_angle}°")

        cv2.imshow("YOLOv8 Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    picam2.stop()
    pwm.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
    print("Resources released.")

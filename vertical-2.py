import cv2
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from ultralytics import YOLO

# ======================= GPIO & Servo Setup =======================
GPIO.setmode(GPIO.BCM)

TILT_SERVO_PIN = 27  # GPIO pin for vertical tilt servo
GPIO.setup(TILT_SERVO_PIN, GPIO.OUT)

tilt_pwm = GPIO.PWM(TILT_SERVO_PIN, 50)  # 50Hz frequency
tilt_pwm.start(0)

# ======================= YOLOv8 & Camera Setup =======================
model = YOLO("best_new.onnx")  # Replace with your trained YOLOv8 model that detects fire

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# ======================= Constants =======================
FRAME_SIZE = (640, 640)  # (width, height)
CAMERA_VERTICAL_FOV = 60  # degrees (adjust based on camera)

# ======================= Helper Functions =======================

def set_tilt_angle(angle):
    duty = 2.5 + (angle / 18)
    duty = max(0.0, min(duty, 12.5))
    tilt_pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    tilt_pwm.ChangeDutyCycle(0)
    print(f"Tilt Servo moved to {angle}° with duty cycle {duty:.2f}%")

def calculate_fire_tilt_angle(y, FRAME_DISPLAY_SIZE=FRAME_SIZE, CAMERA_VERTICAL_FOV=CAMERA_VERTICAL_FOV):
    """
    Inverted tilt logic:
    - If fire is higher (smaller y) -> tilt angle increases (camera tilts down)
    - If fire is lower (larger y) -> tilt angle decreases (camera tilts up)
    """
    origin_y = FRAME_DISPLAY_SIZE[1] // 2
    dy = y - origin_y
    angle_offset = (dy / FRAME_DISPLAY_SIZE[1]) * CAMERA_VERTICAL_FOV
    angle = 90 + angle_offset  # Invert from normal to match fire-tracking requirement
    return int(max(0, min(180, angle)))

# ======================= Main Loop =======================
try:
    last_tilt_angle = None

    while True:
        frame = picam2.capture_array()
        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0])  # Get class ID

                if conf > 0.5:
                    # Assuming class ID 0 is "fire" (adjust if needed)
                    if model.names[cls].lower() == "fire":
                        center_y = (y1 + y2) // 2
                        tilt_angle = calculate_fire_tilt_angle(center_y)

                        if last_tilt_angle is None or abs(tilt_angle - last_tilt_angle) > 2:
                            set_tilt_angle(tilt_angle)
                            last_tilt_angle = tilt_angle

                        # Draw visual feedback
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Fire - Tilt: {tilt_angle}°", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        print(f"🔥 Fire at Y: {center_y} -> Tilt Angle: {tilt_angle}°")

        cv2.imshow("Fire Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    picam2.stop()
    tilt_pwm.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
    print("Resources released.")

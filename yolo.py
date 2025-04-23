import cv2
import math
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2  # Import Picamera2 for Raspberry Pi camera
from ultralytics import YOLO

# Servo configuration
GPIO.setmode(GPIO.BCM)

# Define servo control pins
SERVO_PIN_1 = 17  # First servo pin
SERVO_PIN_2 = 14  # Second servo pin

# Set up the servo pins
GPIO.setup(SERVO_PIN_1, GPIO.OUT)
GPIO.setup(SERVO_PIN_2, GPIO.OUT)

# Create PWM instances with 50Hz frequency
pwm1 = GPIO.PWM(SERVO_PIN_1, 50)  # 50Hz for first servo motor
pwm2 = GPIO.PWM(SERVO_PIN_2, 50)  # 50Hz for second servo motor
pwm1.start(0)  # Start with 0 duty cycle (servo off)
pwm2.start(0)  # Start with 0 duty cycle (servo off)

# Load YOLOv8 model
model = YOLO("best.pt")

# Set up the camera with Picam2
picam2 = Picamera2()  # Initialize Picamera2 instance
picam2.preview_configuration.main.size = (1280, 1280)  # Set the camera resolution
picam2.preview_configuration.main.format = "RGB888"  # Set the camera color format
picam2.preview_configuration.align()  # Align the camera configuration
picam2.configure("preview")  # Configure the camera for preview mode
picam2.start()  # Start the camera

# Get frame size
frame_width = 1280  # Width from the camera resolution
frame_height = 1280  # Height from the camera resolution
origin_x, origin_y = frame_width / 2, frame_height / 2  # Image center

# Object dimension parameters for distance estimation
KNOWN_WIDTH = 1.0  # Replace with actual object width in meters
FOCAL_LENGTH = 1.0  # Replace with actual focal length

def set_angle(angle, pwm):
    """Convert angle (0-180) to duty cycle and move the servo"""
    # Map angle (0-180) to duty cycle (3 to 12%)
    duty = 2.5 + (angle / 18)  # This ensures 0° maps to 2.5% and 180° maps to 12.5%
    
    # Ensure the duty cycle is within the valid range (0 to 100)
    duty = max(0.0, min(duty, 12.5))  # Cap it to the range [0, 12.5] for typical servos
    
    pwm.ChangeDutyCycle(duty)
    time.sleep(2.0)  # Wait for servo to move
    pwm.ChangeDutyCycle(0)  # Stop sending signal to prevent jitter
    print(f"Servo moved to {angle} degrees with duty cycle {duty:.2f}%")

    
def calculate_angle(x, y):
    """ Compute angle θx between detected object and image center """
    dx = x - origin_x
    dy = origin_y - y  # Invert Y since OpenCV's origin is top-left
    theta_x = math.atan2(dy, dx)
    theta_x_degrees = math.degrees(theta_x)

    # Normalize angle (-90 to +90 degrees → 0 to 180 for servo)
    servo_angle = int((theta_x_degrees + 90) * (180 / 180))
    
    return servo_angle

while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()
    
    # Run YOLO inference
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score

            if conf > 0.5:  # Process only confident detections
                bbox_center_x = (x1 + x2) / 2
                bbox_center_y = (y1 + y2) / 2

                # Compute servo angle for both servos
                servo_angle_1 = calculate_angle(bbox_center_x, bbox_center_y)
                servo_angle_2 = calculate_angle(bbox_center_x, bbox_center_y)  # Assuming same angle for both, adjust as needed

                # Move the servos based on the calculated angle
                set_angle(servo_angle_1, pwm1)
                set_angle(servo_angle_2, pwm2)

                # Estimate distance using the bounding box width
                bbox_width = x2 - x1
                estimated_distance = (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width
                real_distance = (estimated_distance * 2300 * 1.5) + 10  # Scale factor

                # Draw detection & annotations
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int(bbox_center_x), int(bbox_center_y)), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Angle: {servo_angle_1}°", (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(frame, f"Distance: {real_distance:.2f} cm", (x1, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                # Print debug info
                print(f"Detection Center: ({bbox_center_x:.2f}, {bbox_center_y:.2f})")
                print(f"Servo Angle 1: {servo_angle_1}°")
                print(f"Servo Angle 2: {servo_angle_2}°")
                print(f"Estimated Distance: {real_distance:.2f} cm")

    # Display frame
    cv2.imshow("YOLOv8 Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the camera and any windows
picam2.stop()
cv2.destroyAllWindows()

# Cleanup PWM and GPIO
pwm1.stop()
pwm2.stop()
GPIO.cleanup()

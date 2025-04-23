import cv2
import math
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")

# Define image path manually
image_path = "/home/mehmood/Desktop/FYP Model/fire.jpeg"  # Change this to your image file path

# Read the input image
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not read the image file.")
    exit()

# Get image dimensions
frame_height, frame_width = image.shape[:2]
origin_x, origin_y = frame_width / 2, frame_height / 2  # Image center

# Object dimension parameters for distance estimation
KNOWN_WIDTH = 1.0  # Replace with actual object width in meters
FOCAL_LENGTH = 1.0  # Replace with actual focal length

def calculate_angle(x, y):
    """ Compute angle ?x between detected object and image center """
    dx = x - origin_x
    dy = origin_y - y  # Invert Y since OpenCV's origin is top-left
    theta_x = math.atan2(dy, dx)
    theta_x_degrees = math.degrees(theta_x)
    
    # Normalize angle (-90 to +90 degrees ? 0 to 180 for servo)
    servo_angle = int((theta_x_degrees + 90) * (180 / 180))
    
    return theta_x_degrees, servo_angle

# Run YOLO inference
results = model(image)

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = box.conf[0].item()  # Confidence score

        if conf > 0.5:  # Process only confident detections
            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2
            bbox_width = x2 - x1

            # Compute angles
            theta_x_degrees, servo_angle = calculate_angle(bbox_center_x, bbox_center_y)

            # Estimate distance using the bounding box width
            estimated_distance = (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width
            real_distance = (estimated_distance * 2300 * 1.5) + 10  # Scale factor

            # Draw detection & annotations
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(image, (int(bbox_center_x), int(bbox_center_y)), 5, (0, 0, 255), -1)
            cv2.putText(image, f"Angle: {theta_x_degrees:.2f}°", (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(image, f"Distance: {real_distance:.2f} cm", (x1, y1 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Print debug info
            print(f"Detection Center: ({bbox_center_x:.2f}, {bbox_center_y:.2f})")
            print(f"Servo Angle: {servo_angle}°")
            print(f"Estimated Distance: {real_distance:.2f} cm")

# Display the image
cv2.imshow("YOLOv11 Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


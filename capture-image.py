import cv2
import time
import os
import datetime
from picamera2 import Picamera2

# Create folder to save images
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

# Initialize the camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

print("Starting webcam... Press 'q' to quit.")

try:
    while True:
        # Capture frame
        frame = picam2.capture_array()

        # Show the frame in a window
        cv2.imshow("Live Camera Feed", frame)

        # Save the frame
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"image_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

        # Wait 2 seconds between frames
        if cv2.waitKey(2000) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera and windows closed.")

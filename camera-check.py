from picamera2 import Picamera2

# Print available cameras
print(Picamera2.global_camera_info())

# Now try initializing the camera
picam2 = Picamera2(camera_num=0)  # Or try camera_num=1 if you have multiple cameras

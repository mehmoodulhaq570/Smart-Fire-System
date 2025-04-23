from picamera2 import Picamera2
import time

picam2 = Picamera2()
picam2.configure("preview")
picam2.start()
time.sleep(2)
picam2.stop()

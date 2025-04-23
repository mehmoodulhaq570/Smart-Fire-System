import RPi.GPIO as GPIO
import time

SMOKE_PIN = 2  # GPIO17

GPIO.setmode(GPIO.BCM)
GPIO.setup(SMOKE_PIN, GPIO.IN)

try:
    while True:
        if GPIO.input(SMOKE_PIN) == GPIO.LOW:
            print("Smoke Detected!")
        else:
            print("Air is Clean.")
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    GPIO.cleanup()

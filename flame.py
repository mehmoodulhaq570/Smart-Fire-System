import RPi.GPIO as GPIO
import time

# Set the GPIO pin
FLAME_SENSOR_PIN = 17  # You can change this to the pin you're using

# Set up GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setup(FLAME_SENSOR_PIN, GPIO.IN)

try:
    print("Flame sensor test. Press Ctrl+C to quit.")
    while True:
        if GPIO.input(FLAME_SENSOR_PIN) == 0:
            print("🔥 Flame detected!")
        else:
            print("✅ No flame.")
        time.sleep(1)

except KeyboardInterrupt:
    print("Exiting program.")

finally:
    GPIO.cleanup()

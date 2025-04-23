import RPi.GPIO as GPIO
import time

# Set GPIO mode
GPIO.setmode(GPIO.BCM)

# Define the servo control pin
SERVO_PIN = 17

# Setup the pin as output
GPIO.setup(SERVO_PIN, GPIO.OUT)

# Create PWM instance with 50Hz frequency
pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz for servo motors
pwm.start(0)  # Start with 0 duty cycle (servo off)

def set_angle(angle):
    """Convert angle (0-180) to duty cycle and move the servo"""
    duty = 3.0 + (angle / 18)  # Formula to map angle to duty cycle
    pwm.ChangeDutyCycle(duty)
    time.sleep(2.0)  # Wait for servo to move
    pwm.ChangeDutyCycle(0)  # Stop sending signal to prevent jitter
    print("Signal sent")

try:
    while True:
        angle = int(input("Enter angle (0-120): "))  # User input
        if 0 <= angle <= 180:
            set_angle(angle)
        else:
            print("Please enter a valid angle between 0 and 180.")

except KeyboardInterrupt:
    print("\nExiting program...")

finally:
    pwm.stop()  # Stop PWM
    GPIO.cleanup()  # Cleanup GPIO
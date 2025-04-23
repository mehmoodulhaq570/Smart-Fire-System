import RPi.GPIO as GPIO
import time

# Set GPIO mode
GPIO.setmode(GPIO.BCM)

# Define the servo control pins
SERVO_PIN_1 = 17  # First servo
SERVO_PIN_2 = 27  # Second servo

# Setup the pins as output
GPIO.setup(SERVO_PIN_1, GPIO.OUT)
GPIO.setup(SERVO_PIN_2, GPIO.OUT)

# Create PWM instances with 50Hz frequency
pwm1 = GPIO.PWM(SERVO_PIN_1, 50)  # 50Hz for servo motors
pwm2 = GPIO.PWM(SERVO_PIN_2, 50)

pwm1.start(0)  # Start with 0 duty cycle (servo off)
pwm2.start(0)

def set_angle(angle):
    """Convert angle (0-180) to duty cycle and move both servos"""
    duty = 3.0 + (angle / 18)  # Formula to map angle to duty cycle
    pwm1.ChangeDutyCycle(duty)
    pwm2.ChangeDutyCycle(duty)
    time.sleep(2.0)  # Wait for servos to move
    pwm1.ChangeDutyCycle(0)  # Stop sending signal to prevent jitter
    pwm2.ChangeDutyCycle(0)
    print("Signal sent to both servos")

try:
    while True:
        angle = int(input("Enter angle for both servos (0-180): "))  # User input for both servos
        
        if 0 <= angle <= 180:
            set_angle(angle)
        else:
            print("Please enter a valid angle between 0 and 180.")

except KeyboardInterrupt:
    print("\nExiting program...")

finally:
    pwm1.stop()  # Stop PWM for servo 1
    pwm2.stop()  # Stop PWM for servo 2
    GPIO.cleanup()  # Cleanup GPIO
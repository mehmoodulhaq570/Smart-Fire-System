import RPi.GPIO as GPIO
import time

# Pin Definitions
ENA = 18  # Pin for PWM (Speed control)
IN1 = 23  # Direction control 1
IN2 = 24  # Direction control 2

# Setup GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)

# Set IN1 and IN2 to control one direction
GPIO.output(IN1, GPIO.HIGH)  # Keep IN1 high to maintain forward direction
GPIO.output(IN2, GPIO.LOW)   # Keep IN2 low to maintain forward direction

# Set up PWM for speed control on ENA pin
pwm = GPIO.PWM(ENA, 1000)  # 1 kHz frequency for PWM
pwm.start(0)  # Start PWM with 0% duty cycle (off)

try:
    while True:
        # Prompt user to enter a speed value between 0 and 100
        user_input = input("Enter speed (0 to 100): ")
        
        # Validate user input
        if user_input.isdigit():
            speed = int(user_input)
            
            if 0 <= speed <= 100:
                pwm.ChangeDutyCycle(speed)
                print(f"Speed set to: {speed}%")
            else:
                print("Please enter a value between 0 and 100.")
        else:
            print("Invalid input. Please enter a number between 0 and 100.")
        
        time.sleep(1)  # Small delay to prevent overwhelming the user input prompt

except KeyboardInterrupt:
    print("Program stopped by user")

finally:
    pwm.stop()
    GPIO.cleanup()

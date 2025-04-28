import RPi.GPIO as GPIO
import time

# GPIO setup
GPIO.setmode(GPIO.BCM)
BUZZER_PIN = 6
GPIO.setup(BUZZER_PIN, GPIO.OUT)

buzzer_state = False  # Track whether the buzzer is on or off

try:
    print("Type 'on' to turn on the buzzer, 'off' to turn it off, 'exit' to quit.")
    while True:
        user_input = input("Enter command: ").strip().lower()

        if user_input == "on":
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            buzzer_state = True
            print("Buzzer is ON")

        elif user_input == "off":
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            buzzer_state = False
            print("Buzzer is OFF")

        elif user_input == "exit":
            print("Exiting...")
            break

        else:
            print("Invalid input. Type 'on', 'off', or 'exit'.")

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    GPIO.output(BUZZER_PIN, GPIO.LOW)  # Ensure buzzer is off
    GPIO.cleanup()

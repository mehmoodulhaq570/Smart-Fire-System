import RPi.GPIO as GPIO
import time
from RPLCD.i2c import CharLCD

# Pin Configuration
mq5_pin = 26
flame_pin = 16

# LCD Configuration
lcd = CharLCD('PCF8574', 0x27)  # I2C address 0x27 usually, check yours with sudo i2cdetect -y 1

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(mq5_pin, GPIO.IN)
GPIO.setup(flame_pin, GPIO.IN)

try:
    while True:
        gas_status = GPIO.input(mq5_pin)
        flame_status = GPIO.input(flame_pin)

        lcd.clear()

        if gas_status == GPIO.LOW:
            lcd.write_string("Gas: Detected")
        else:
            lcd.write_string("Gas: Normal")

        lcd.crlf()  # Move to next line

        if flame_status == GPIO.LOW:
            lcd.write_string("Flame: Detected")
        else:
            lcd.write_string("Flame: Safe")

        time.sleep(1)

except KeyboardInterrupt:
    GPIO.cleanup()
    lcd.clear()
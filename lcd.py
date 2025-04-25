from RPLCD.i2c import CharLCD
from time import sleep

lcd = CharLCD('PCF8574', 0x27)  # Use your I2C address here

lcd.write_string("Hello, Talal!")
sleep(20)
lcd.clear()
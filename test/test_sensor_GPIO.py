import RPi.GPIO as GPIO  # rpi-lgpio provides this exact interface
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.IN)

prev_input = 0

try:
    while True:
        input = GPIO.input(4)
        if (not prev_input) and input:
            print("Under Pressure")
        prev_input = input
        time.sleep(0.10)
except KeyboardInterrupt:
    GPIO.cleanup()
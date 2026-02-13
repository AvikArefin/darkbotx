import time 
import board 
import busio

import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# Initialize I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

# Create the ADS object
ads = ADS.ADS1115(i2c)
ads.gain = 1

# CORRECTED LINE: 

# Since 'ADS' refers to the ads1115 module, 
# use the integer 0 directly which the library also accepts.
chan = AnalogIn(ads, 0) 

while True:
    print(f"Flex / Force Sensor Voltage: {chan.voltage:.3f}V")
    time.sleep(1)
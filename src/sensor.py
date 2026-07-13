import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

from enum import IntEnum


class Pin(IntEnum):
    """
    Sensor Pin

    Don't use GRIPPER_RIGHT that one is already mirrored from gripper left
    """

    LEFT_FLEX = 0
    RIGHT_FLEX = 1
    LEFT_FORCE = 2
    RIGHT_FORCE = 3


class SensorError(Exception):
    """Raised when the sensor fails to initialize or read."""

    pass


# TODO: Upon initialization, automatically set the current voltages as the defaults,
# and LESS?? or MORE??? than that initial value,
# should indicate that the we are getting `force` i.e. data on that sensor
# the v_min and v_max needs to be dynamic
class Sensor:
    # def __init__(self, i2c_bus: busio.I2C, v_min: float, v_max: float):
    def __init__(self, channels: int, i2c: busio.I2C) -> None:
        try:
            self.channels = channels
            self.ads = ADS.ADS1115(i2c)
            self.ads.gain = 1.0

            # Create a list of all ports. Ex: 4 (P0, P1, P2, P3)
            self.ports = [AnalogIn(self.ads, i) for i in range(self.channels)]

            self._initialized = True
            self.init_volts = self.get_all_voltages()


        except Exception as e:
            self._initialized = False
            raise SensorError(f"Failed to initialize ADS1115: {e}") from e

    def _assert_ready(self):
        if not self._initialized:
            raise SensorError("Sensor is not initialized.")

    def get_voltage(self, pin: int) -> float:
        """Reads voltage from a specific pin (0-3)."""
        self._assert_ready()
        if not (0 <= pin < self.channels):
            raise ValueError(f"Pin index must be between 0 and {self.channels - 1}.")

        try:
            v = self.ports[pin].voltage
        except Exception as e:
            raise SensorError(f"Failed to read voltage on pin {pin}: {e}") from e
        return v

    def get_all_voltages(self) -> list[float]:
        """Returns a list of voltages from all 4 pins."""
        return [self.get_voltage(i) for i in range(self.channels)]


if __name__ == "__main__":
    try:
        import board
        _i2c_bus = busio.I2C(board.SCL, board.SDA)
        sensor = Sensor(channels=4, i2c=_i2c_bus)
        print(sensor.get_all_voltages())

    finally:
        _i2c_bus.deinit()
import logging
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

logger = logging.getLogger(__name__)


class SensorError(Exception):
    """Raised when the sensor fails to initialize or read."""

    pass


class Sensor:
    def __init__(self, i2c_bus, gain: int = 1, v_min: float = 1.0, v_max: float = 3.2):
        try:
            if v_min >= v_max:
                self._initialized = False
                raise SensorError("v_min can not be greater than or equal to v_max!")

            self.V_MIN: float = v_min
            self.V_MAX: float = v_max
            self.ads = ADS.ADS1115(i2c_bus)
            self.ads.gain = gain

            # Create a list of all 4 channels (P0, P1, P2, P3)
            self.channels = [AnalogIn(self.ads, i) for i in range(4)]

            self._initialized = True
        except Exception as e:
            self._initialized = False
            raise SensorError(f"Failed to initialize ADS1115: {e}") from e

    def _assert_ready(self):
        if not self._initialized:
            raise SensorError("Sensor is not initialized.")

    def get_voltage(self, pin: int) -> float:
        """Reads voltage from a specific pin (0-3)."""
        self._assert_ready()
        if not (0 <= pin <= 3):
            raise ValueError("Pin index must be between 0 and 3.")

        try:
            v = self.channels[pin].voltage
        except Exception as e:
            raise SensorError(f"Failed to read voltage on pin {pin}: {e}") from e

        if not (self.V_MIN <= v <= self.V_MAX):
            logger.warning(
                "Pin %d: Voltage %.3fV out of expected range [%.1f, %.1f]",
                pin,
                v,
                self.V_MIN,
                self.V_MAX,
            )
        return v

    def get_all_voltages(self) -> list:
        """Returns a list of voltages from all 4 pins."""
        return [self.get_voltage(i) for i in range(4)]

    def get_mapped_value(self, pin: int, angle_min: float, angle_max: float) -> float:
        """Maps a specific pin's voltage to a range [angle_min, angle_max]."""
        if angle_min >= angle_max:
            raise ValueError(
                f"angle_min ({angle_min}) must be less than angle_max ({angle_max})"
            )

        v = self.get_voltage(pin)
        v_clamped = max(self.V_MIN, min(v, self.V_MAX))
        mapped = (v_clamped - self.V_MIN) * (angle_max - angle_min) / (
            self.V_MAX - self.V_MIN
        ) + angle_min

        logger.debug("pin=%d voltage=%.3fV → mapped=%.2f", pin, v, mapped)
        return mapped

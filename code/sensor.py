import logging
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

logger = logging.getLogger(__name__)


class SensorError(Exception):
    """Raised when the sensor fails to initialize or read."""
    pass


class Sensor:
    # TODO: Turn it into 4 sensor sensing 2 from finger + 1 motor feedback + 1 Extra Sensor
    def __init__(self, i2c_bus, gain: int = 1, v_min: float = 1.0, v_max: float = 3.2):
        try:
            if v_min >= v_max:
                self._initialized = False
                raise SensorError("v_min can not be greater than or equal to v_max!")
            self.V_MIN: float = v_min
            self.V_MAX: float = v_max
            self.ads = ADS.ADS1115(i2c_bus)
            self.ads.gain = gain
            self.chan = AnalogIn(self.ads, 0)
            self._initialized = True
        except Exception as e:
            self._initialized = False
            raise SensorError(f"Failed to initialize ADS1115: {e}") from e

    def _assert_ready(self):
        if not self._initialized:
            raise SensorError("Sensor is not initialized.")

    def get_voltage(self) -> float:
        self._assert_ready()
        try:
            v = self.chan.voltage
        except Exception as e:
            raise SensorError(f"Failed to read voltage: {e}") from e

        if not (self.V_MIN <= v <= self.V_MAX):
            logger.warning(
                "Voltage %.3fV out of expected range [%.1f, %.1f]",
                v,
                self.V_MIN,
                self.V_MAX,
            )

        return v

    def get_mapped_value(self, angle_min: float, angle_max: float) -> float:
        """
        Maps the sensor voltage to [out_min, out_max].
        Voltage is clamped to [V_MIN, V_MAX] before mapping.
        Raises SensorError on read failure or invalid output range.
        """
        if angle_min >= angle_max:
            raise ValueError(
                f"out_min ({angle_min}) must be less than out_max ({angle_max})"
            )

        v = self.get_voltage()  # already guarded
        v_clamped = max(self.V_MIN, min(v, self.V_MAX))
        mapped = (v_clamped - self.V_MIN) * (angle_max - angle_min) / (
            self.V_MAX - self.V_MIN
        ) + angle_min

        logger.debug(
            "voltage=%.3fV → mapped=%.2f (range [%s, %s])", v, mapped, angle_min, angle_max
        )
        return mapped

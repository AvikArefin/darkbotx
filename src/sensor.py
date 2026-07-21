from collections import deque
from enum import IntEnum
import sys
import threading
from typing import Any

import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import busio


class Pin(IntEnum):
    """
    Sensor Pin

    Don't use GRIPPER_RIGHT that one is already mirrored from gripper left
    """

    LEFT_FLEX = 0
    LEFT_FORCE = 1
    RIGHT_FLEX = 2
    RIGHT_FORCE = 3


class SensorError(Exception):
    """Raised when the sensor fails to initialize or read."""

    pass


class Sensor:
    def __init__(self, channels: int, i2c: busio.I2C, lock: Any = None) -> None:
        try:
            self.channels = channels
            self.lock = lock
            self.ads = ADS.ADS1115(i2c)
            self.ads.gain = 1.0

            # Create a list of all ports. Ex: 4 (P0, P1, P2, P3)
            self.ports = [AnalogIn(self.ads, i) for i in range(self.channels)]

            self._initialized = True
            # Warmup First to get correct values
            self.get_all_voltages()
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
            if self.lock:
                with self.lock:
                    v = self.ports[pin].voltage
            else:
                v = self.ports[pin].voltage
        except Exception as e:
            raise SensorError(f"Failed to read voltage on pin {pin}: {e}") from e
        return round(v, 3)

    def get_all_voltages(self) -> list[float]:
        """Returns a list of voltages from all 4 pins."""
        return [self.get_voltage(i) for i in range(self.channels)]


def format_voltages(volts: list[float]) -> str:
    """Formats a list of voltages as a string with 3 decimal places for each value."""
    return "[" + ", ".join(f"{v:.3f}" for v in volts) + "]"


def calc_avg_variation(history: list[list[float]]) -> list[float]:
    """Calculates the average voltage variation between consecutive readings for each pin."""
    if len(history) < 2:
        return [0.0] * (len(history[0]) if history else 4)
    n_pins = len(history[0])
    n_diffs = len(history) - 1
    avg_var = []
    for pin in range(n_pins):
        diff_sum = sum(abs(history[i][pin] - history[i - 1][pin]) for i in range(1, len(history)))
        avg_var.append(round(diff_sum / n_diffs, 3))
    return avg_var


class SensorMonitor:
    """Tracks history and variation stats for a Sensor, and handles formatted output."""

    def __init__(self, sensor: Sensor, history_len: int = 5) -> None:
        self.sensor = sensor
        self.history_len = history_len
        self.history: deque[list[float]] = deque(maxlen=history_len)
        self.min_volts = list(sensor.init_volts)
        self.max_volts = list(sensor.init_volts)

    def update(self) -> list[float]:
        """Reads current voltages and updates history, min, and max values."""
        volts = self.sensor.get_all_voltages()
        self.history.append(volts)
        for i, v in enumerate(volts):
            if v < self.min_volts[i]:
                self.min_volts[i] = v
            if v > self.max_volts[i]:
                self.max_volts[i] = v
        return volts

    def get_lines(self) -> list[str]:
        """Generates formatted lines for sensor stats and history."""
        volts = self.history[-1] if self.history else [0.0] * self.sensor.channels
        avg_var = calc_avg_variation(list(self.history))
        max_var = [round(self.max_volts[i] - self.min_volts[i], 3) for i in range(len(self.min_volts))]
        diff_volts = [round(volts[i] - self.sensor.init_volts[i], 3) for i in range(len(volts))]

        lines = [
            f"Init: {format_voltages(self.sensor.init_volts)}",
            f"Diff: {format_voltages(diff_volts)}",
            f"Avg variation: {format_voltages(avg_var)}",
            f"Max variation: {format_voltages(max_var)}",
            f"Min: {format_voltages(self.min_volts)}",
            f"Max: {format_voltages(self.max_volts)}",
        ]
        for r in self.history:
            lines.append(format_voltages(r))
        for _ in range(self.history_len - len(self.history)):
            lines.append("")
        return lines

    def print_status(self, overlay: bool = False) -> None:
        """Prints current sensor stats. If overlay is True, renders at top left of terminal."""
        lines = self.get_lines()
        if overlay:
            sys.stdout.write("\033[s\033[1;1H")
            for line in lines:
                sys.stdout.write(f"{line}\033[K\n")
            sys.stdout.write("-" * 50 + "\033[K\n")
            sys.stdout.write("\033[u")
            sys.stdout.flush()
        else:
            sys.stdout.write("\033[H\033[J")
            for line in lines:
                sys.stdout.write(f"{line}\n")
            sys.stdout.flush()

    def update_and_print(self, overlay: bool = False) -> None:
        """Reads voltages, updates history, and prints current status."""
        self.update()
        self.print_status(overlay=overlay)


if __name__ == "__main__":
    _i2c_bus = None
    try:
        import time
        import board

        _i2c_bus = busio.I2C(board.SCL, board.SDA)
        sensor = Sensor(channels=4, i2c=_i2c_bus)
        monitor = SensorMonitor(sensor)

        while True:
            monitor.update_and_print()
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped printing voltages.")
    finally:
        if _i2c_bus is not None:
            _i2c_bus.deinit()
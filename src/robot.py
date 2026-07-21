from enum import IntEnum
import time
import threading

from dataclasses import dataclass

from sensor import Sensor, Pin
from adafruit_servokit import ServoKit
import busio
import board


class RJoint(IntEnum):
    """
    Hiwonder Real Joints

    Don't use GRIPPER_RIGHT that one is already mirrored from gripper left
    """

    BASE = 5
    SHOULDER = 4
    ELBOW = 3
    WRIST = 2
    WRIST_ROLL = 1
    GRIPPER = 15
    # GRIPPER_RIGHT = 6

WIDE_GRIP = 245

MIN_LENGTH = 1.4
MAX_LENGTH = 6.4

@dataclass(frozen=True)
class ServoConfig:
    min_pulse: int
    max_pulse: int
    angle_range: int = 180  # Most joints are 180, so it serves as the default


class RobotArm:
    SERVO_CONFIG: dict[int, ServoConfig] = {
        0: ServoConfig(490, 2660, angle_range=270),  # Gripper
        1: ServoConfig(450, 2650),  # Wrist Rot (180 max_angle)
        2: ServoConfig(500, 2650),  # Wrist  (180 max_angle)
        3: ServoConfig(500, 2700),  # Elbow (180 max_angle)
        4: ServoConfig(500, 2600),  # Shoulder (180 max_angle)
        5: ServoConfig(500, 2600),  # Base (180 max_angle)
        15: ServoConfig(
            500, 2650, angle_range=270
        ),  # Gripper (same as 0, used for testing)
    }

    # Going till 270 would cause fingers to go out of teeth.
    # which means, the fingers would come out. So, we are going till [WIDE_GRIP].
    # 15 is simply a dummy used for testing similar behaviour as 0 i.e. gripper fingers

    HOME_POSITION: dict[int, int] = {
        0: WIDE_GRIP,
        1: 0,
        2: 90,
        3: 90,
        4: 90,
        5: 105,
        15: WIDE_GRIP,
    }

    PUT_POSITION: dict[int, int] = {
        # 0: 10,
        # 1: 0,
        2: 0,
        3: 50,
        4: 115,
        5: 105,
        # 15: 10,
    }

    LIFT_POSITION: dict[int, int] = {
        # 0: 10,
        # 1: 0,
        2: 20,
        3: 90,
        4: 60,
        5: 105,
        # 15: 10,
    }

    GRAB_POSITION: dict[int, int] = {
        0: WIDE_GRIP,
        1: 0,
        2: 0,
        3: 50,
        4: 115,
        5: 105,
        15: WIDE_GRIP,
    }

    def __init__(self):
        self._i2c_bus = busio.I2C(board.SCL, board.SDA)
        self.i2c_lock = threading.Lock()
        self.sensor = Sensor(channels=4, i2c=self._i2c_bus, lock=self.i2c_lock)
        self.kit = ServoKit(channels=16, i2c=self._i2c_bus)

        self.current_angles: dict[int, float] = {}
        self._initialized = False
        self._setup_servos()

    def deinit(self):
        """Releases the I2C bus when shutting down."""
        self._i2c_bus.deinit()

    def set_servo_angle(self, channel: int, angle: float | int) -> None:
        """Sets a servo angle in a thread-safe manner."""
        with self.i2c_lock:
            self.kit.servo[channel].angle = int(angle)

    def _setup_servos(self) -> None:
        """Initializes servos with individual pulse width ranges."""
        try:
            RED = "\033[91m"
            RESET = "\033[0m"
            BOLD = "\033[1m"

            print(
                f"\n{RED}{BOLD}WARNING: Bot might move abruptly to home position!!!{RESET}"
            )

            for i in range(3, 0, -1):
                print(f"{RED}{i}...{RESET}")
                time.sleep(1)

            for ch, servo in self.SERVO_CONFIG.items():
                with self.i2c_lock:
                    self.kit.servo[ch].set_pulse_width_range(
                        servo.min_pulse, servo.max_pulse
                    )
                    self.kit.servo[ch].actuation_range = servo.angle_range

                home_angle = self.HOME_POSITION[ch]
                self.set_servo_angle(ch, home_angle)
                self.current_angles[ch] = float(home_angle)

            self._initialized = True
            print("Initialization Complete. Robot is Armed\n")
        except Exception as e:
            print(f"Servo setup failed: {e}")
            raise

    def move_smooth(
        self,
        channel: int,
        target_angle: float,
        delay: float = 0.01,
        step_size: float = 1.0,
    ) -> bool:
        """Moves a single servo smoothly to a target angle."""
        if not self._initialized:
            print("Arm uninitialized")
            return False

        if channel not in self.SERVO_CONFIG:
            print(f"Invalid channel {channel}")
            return False

        max_angle = self.SERVO_CONFIG[channel].angle_range
        safe_target = max(0.0, min(target_angle, max_angle))

        if safe_target != target_angle:
            print(f"Ch{channel}: Target {target_angle}° clamped to {safe_target}°")

        current = self.current_angles.get(channel, float(self.HOME_POSITION[channel]))

        if abs(current - safe_target) < 0.1:
            return True

        direction = 1 if safe_target > current else -1

        while abs(safe_target - current) > step_size:
            current += step_size * direction
            self.set_servo_angle(channel, current)
            time.sleep(delay)

        self.set_servo_angle(channel, safe_target)
        self.current_angles[channel] = safe_target
        return True

    def move_all_smooth(
        self,
        target_positions: dict[int, int],
        delay: float = 0.01,
        max_step: float = 1.0,
    ) -> None:
        """Simultaneously moves multiple servos to their target angles proportionally."""
        if not self._initialized:
            print("Cannot move: Arm uninitialized.")
            return

        distances = {}
        start_angles = {}

        # 1. Calculate the required travel distance for each valid servo
        for ch, target in target_positions.items():
            if ch not in self.SERVO_CONFIG:
                continue

            max_angle = self.SERVO_CONFIG[ch].angle_range
            safe_target = max(0.0, min(float(target), float(max_angle)))

            if safe_target != target:
                print(f"Ch{ch}: Target {target}° clamped to {safe_target}°")

            current = self.current_angles.get(ch, float(self.HOME_POSITION[ch]))
            start_angles[ch] = current
            distances[ch] = safe_target - current

        if not distances:
            return

        # 2. Find the maximum distance any single servo has to travel
        max_dist: float | int = max((abs(d) for d in distances.values()), default=0)

        if max_dist < 0.1:
            return  # All servos are already at their target positions

        # 3. Determine the number of steps based on the largest movement
        steps = int(max_dist / max_step)
        if steps == 0:
            steps = 1

        # 4. Move all servos incrementally
        for step in range(1, steps + 1):
            for ch in distances.keys():
                # Linear interpolation: move proportionally based on the current step
                current_target = start_angles[ch] + (distances[ch] * (step / steps))
                self.set_servo_angle(ch, current_target)
                self.current_angles[ch] = current_target

            # Pause once per step to control overall speed
            time.sleep(delay)

        # 5. Lock in the exact final target angles to account for float math drift
        for ch in distances.keys():
            safe_target = start_angles[ch] + distances[ch]
            self.set_servo_angle(ch, safe_target)
            self.current_angles[ch] = safe_target

    def go_home_smooth(self, delay: float = 0.01, max_step: float = 1.0) -> None:
        """Smoothly and simultaneously moves all servos to HOME."""
        print("\nReturning to HOME position smoothly...")
        self.move_all_smooth(self.HOME_POSITION, delay, max_step)
        print("Arm is now at HOME.")

    def go_lift_smooth(self, delay: float = 0.01, max_step: float = 1.0) -> None:
        """Smoothly and simultaneously moves all servos to LIFT."""
        print("\nReturning to LIFT position smoothly...")
        self.move_all_smooth(self.LIFT_POSITION, delay, max_step)
        print("Arm is now at LIFT.")

    def go_put_smooth(self, delay: float = 0.01, max_step: float = 1.0) -> None:
        """Smoothly and simultaneously moves all servos to PUT position."""
        print("\nReturning to PUT position smoothly...")
        self.move_all_smooth(self.PUT_POSITION, delay, max_step)
        print("Arm is now at PUT.")

    def go_grab_smooth(self, delay: float = 0.01, max_step: float = 1.0) -> None:
        """Smoothly and simultaneously moves all servos to GRAB position."""
        print("\nMoving to GRAB position smoothly...")
        self.move_all_smooth(self.GRAB_POSITION, delay, max_step)
        print("Arm is now in GRAB position.")

    def angle2length(self, length_in_angle: float)-> float:
        """
        length
        min = 1.2 cm 
        max = 6.2 cm [for WIDE_GRIP == 245]
        
        angle 
        min = 0
        max = WIDE_GRIP
        """
        # TODO: NEEDS TUNING
        return MIN_LENGTH + length_in_angle * (MAX_LENGTH - MIN_LENGTH) / WIDE_GRIP

    def gripper_close_till_obstacle(self, delay: float = 0.01) -> float:
        channel = RJoint.GRIPPER
        current_angle = self.current_angles.get(
            channel, float(self.HOME_POSITION.get(channel, WIDE_GRIP))
        )
        
        # Warmup
        _ = self.sensor.get_all_voltages()

        init_left = self.sensor.get_voltage(pin=Pin.LEFT_FLEX)
        init_right = self.sensor.get_voltage(pin=Pin.RIGHT_FLEX)

        while current_angle > 0:
            left = self.sensor.get_voltage(pin=Pin.LEFT_FLEX)
            right = self.sensor.get_voltage(pin=Pin.RIGHT_FLEX)

            # Stop when BOTH sensors drop below the threshold (indicating solid contact)
            # TODO: we would need to decide on `or` and `and` and also when the data capture should occur.
            # For test purpose, we will be only checking the left flex as the rest of the values are floating (constantly changing), 
            # when non connected.
            
            if ((left - init_left) > 0.1 and (right - init_right) > 0.11) or ((left - init_left) > 0.15 or (right - init_right) > 0.15):
                break


            # Incrementally close towards 0 degrees
            current_angle = max(0.0, current_angle - 1.0)
            self.set_servo_angle(channel, current_angle)
            self.current_angles[channel] = current_angle

            time.sleep(delay)
        
        return current_angle

    def gripper_open(self):
        self.move_smooth(channel=RJoint.GRIPPER, target_angle=WIDE_GRIP)
        pass

    def scan(self, slice: int) -> tuple[list[tuple[float, float, str]], tuple[float, float]]:
        wrist_angle = 0
        increment_value: float = 180 / slice
        scan_results: list[tuple[float, float, str]] = []
        emergency: tuple[float, float] = (0.0, float('inf'))
        for i in range(slice):
            self.move_smooth(channel=RJoint.WRIST_ROLL, target_angle=wrist_angle)
            gripper_angle = self.gripper_close_till_obstacle()
            length = self.angle2length(gripper_angle)
            self.gripper_open()
            scan_results.append((wrist_angle, length, "both"))
            if gripper_angle < emergency[1]:
                emergency = (wrist_angle, gripper_angle)
            wrist_angle += increment_value

        return scan_results, emergency

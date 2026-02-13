import time
import logging
from adafruit_servokit import ServoKit

logger = logging.getLogger(__name__)

class RobotArm:
    # Pulse width (min, max) applied to all servos
    PULSE: tuple[int, int] = (500, 2500)
    
    # Channel: Max Actuation Range (degrees)
    SERVO_CONFIG: dict[int, int] = {
        0: 270,  # Gripper
        1: 180,  # Joint 1
        2: 180,  # Joint 2
        3: 180,  # Joint 3
        4: 180,  # Joint 4
        5: 180,  # Joint 5
    }

    # Channel: Target Home Angle
    HOME_POSITIONS: dict[int, int] = {
        0: 270,
        1: 90,
        2: 90,
        3: 90,
        4: 90,
        5: 90,
    }

    def __init__(self, i2c_bus):
        self.kit = ServoKit(channels=16, i2c=i2c_bus)
        self._initialized = False
        self._setup_servos()

    def _setup_servos(self) -> None:
        """Initializes servos using the class constants."""
        try:
            for ch, max_range in self.SERVO_CONFIG.items():
                # Unpack the PULSE tuple using *
                self.kit.servo[ch].set_pulse_width_range(*self.PULSE)
                self.kit.servo[ch].actuation_range = max_range
                
            self._initialized = True
        except Exception as e:
            logger.error(f"Servo setup failed: {e}")
            self._initialized = False
            raise

    def home_pos(self) -> bool:
        """Moves all servos to home positions defined in HOME_POSITIONS."""
        if not self._initialized:
            logger.error("Cannot home: arm not initialized.")
            return False

        RED, BOLD, RESET = "\033[91m", "\033[1m", "\033[0m"
        print(f"\n{RED}{BOLD}WARNING: CLEAR OUT HOME POSITION!!!{RESET}")
        for i in range(3, 0, -1):
            print(f"{RED}{i}...{RESET}")
            time.sleep(1)
        print(f"{BOLD}ACTION!{RESET}\n")

        all_ok = True
        for channel, angle in self.HOME_POSITIONS.items():
            if not self.move_servo(channel, angle):
                logger.error(f"Failed to home channel {channel}.")
                all_ok = False

        if all_ok:
            logger.info("Darkbot in HOME POSITION.")
        return all_ok

    def move_servo(self, channel: int, angle: int) -> bool:
        """Moves a servo with safety clamping based on SERVO_CONFIG range."""
        if not self._initialized or channel not in self.SERVO_CONFIG:
            return False

        try:
            # Get max range from config for safety clamping
            max_range = self.SERVO_CONFIG[channel]
            safe_angle = int(max(0, min(int(angle), int(max_range))))   # For floating-point numbers, this truncates towards zero.

            if safe_angle != angle:
                logger.warning(f"Ch{channel}: {angle}° clamped to {safe_angle}°")

            self.kit.servo[channel].angle = safe_angle
            return True

        except Exception as e:
            logger.error(f"Failed to move servo {channel}: {e}")
            return False
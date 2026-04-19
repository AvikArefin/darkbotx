import time
import logging
from dataclasses import dataclass
from adafruit_servokit import ServoKit

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ServoConfig:
    min_pulse: int
    max_pulse: int
    max_angle: int = 180  # Most joints are 180, so it serves as the default

class RobotArm:
    SERVO_CONFIG: dict[int, ServoConfig] = {
        0: ServoConfig(490, 2660, max_angle=270),  # Gripper
        1: ServoConfig(450, 2650), 
        2: ServoConfig(500, 2650),                 # Joint 2 (180 max_angle)
        3: ServoConfig(500, 2700),                 # Joint 3 (180 max_angle)
        4: ServoConfig(500, 2600),                 # Joint 4 (180 max_angle)
        5: ServoConfig(500, 2600),                 # Joint 5 (180 max_angle)
    }

    HOME_POSITIONS: dict[int, int] = {
        0: 270, 1: 90, 2: 90, 3: 90, 4: 90, 5: 90,
    }

    def __init__(self, i2c_bus=None):
        if i2c_bus:
            self.kit = ServoKit(channels=16, i2c=i2c_bus)
        else:
            self.kit = ServoKit(channels=16)
            
        self.current_angles: dict[int, float] = {}
        self._initialized = False
        self._setup_servos()

    def _setup_servos(self) -> None:
        """Initializes servos with individual pulse width ranges."""
        try:
            RED   = "\033[91m"
            RESET = "\033[0m"
            BOLD  = "\033[1m"

            print(f"\n{RED}{BOLD}WARNING: Bot might move abruptly to home position!!!{RESET}")

            for i in range(3, 0, -1):
                print(f"{RED}{i}...{RESET}")
                time.sleep(1)
                
            for ch, servo in self.SERVO_CONFIG.items():
                # Apply the properties directly from the dataclass
                self.kit.servo[ch].set_pulse_width_range(servo.min_pulse, servo.max_pulse)
                self.kit.servo[ch].actuation_range = servo.max_angle
                
                home_angle = self.HOME_POSITIONS.get(ch, 90)
                self.kit.servo[ch].angle = home_angle
                self.current_angles[ch] = float(home_angle)
                
            self._initialized = True
            print("Initialization Complete. Arm is ready.\n")
        except Exception as e:
            logger.error(f"Servo setup failed: {e}")
            raise

    def move_smooth(self, channel: int, target_angle: float, delay: float = 0.01, step_size: float = 1.0) -> bool:
        if not self._initialized or channel not in self.SERVO_CONFIG:
            logger.error(f"Cannot move channel {channel}: Arm uninitialized or invalid channel.")
            return False

        # Access max_angle cleanly from the dataclass instance
        max_angle = self.SERVO_CONFIG[channel].max_angle
        safe_target = max(0.0, min(float(target_angle), float(max_angle)))

        if safe_target != target_angle:
            logger.warning(f"Ch{channel}: Target {target_angle}° clamped to {safe_target}°")

        current = self.current_angles.get(channel, float(self.HOME_POSITIONS.get(channel, 90)))
        
        if abs(current - safe_target) < 0.1:
            return True
            
        direction = 1 if safe_target > current else -1
        
        while abs(safe_target - current) > step_size:
            current += (step_size * direction)
            self.kit.servo[channel].angle = current # type: ignore
            time.sleep(delay)
        
        self.kit.servo[channel].angle = safe_target # type: ignore
        self.current_angles[channel] = safe_target
        return True
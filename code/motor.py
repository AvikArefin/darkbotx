import time
import logging
from adafruit_servokit import ServoKit

logger = logging.getLogger(__name__)

class RobotArm:
    # Default pulse used if a specific one isn't defined in the new config
    DEFAULT_PULSE: tuple[int, int] = (500, 2500)
    
    # Updated SERVO_CONFIG to include (max_angle, (min_pulse, max_pulse))
    # Adjust the pulse values for Channel 1 (or whichever is the DSS-M15S)
    # until 180 degrees on your protractor matches 180 in code.
    SERVO_CONFIG: dict[int, tuple[int, tuple[int, int]]] = {
        0: (270, (490, 2660)),  # Gripper
        1: (270, (450, 2650)),  # Joint 1: Adjusted for 270 motor undershoot
        2: (180, (500, 2650)),  # Joint 2
        3: (180, (500, 2700)),  # Joint 3
        4: (180, (500, 2600)),  # Joint 4
        5: (180, (500, 2600)),  # Joint 5
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
                
            for ch, config in self.SERVO_CONFIG.items():
                max_range, pulse_range = config
                
                # Apply the individual pulse width for this specific channel
                self.kit.servo[ch].set_pulse_width_range(*pulse_range)
                self.kit.servo[ch].actuation_range = max_range
                
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

        # Accessing the max_angle from the new tuple structure
        max_angle = self.SERVO_CONFIG[channel][0]
        safe_target = max(0.0, min(float(target_angle), float(max_angle)))

        if safe_target != target_angle:
            logger.warning(f"Ch{channel}: Target {target_angle}° clamped to {safe_target}°")

        current = self.current_angles.get(channel, float(self.HOME_POSITIONS.get(channel, 90)))
        
        if abs(current - safe_target) < 0.1:
            return True
            
        direction = 1 if safe_target > current else -1
        
        while abs(safe_target - current) > step_size:
            current += (step_size * direction)
            self.kit.servo[channel].angle = current
            time.sleep(delay)
        
        self.kit.servo[channel].angle = safe_target
        self.current_angles[channel] = safe_target
        return True
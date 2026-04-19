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

    HOME_POSITION: dict[int, int] = {
        0: 270, 1: 90, 2: 90, 3: 90, 4: 90, 5: 105,
    }

    GRAB_POSITION: dict[int, int] = {
        0: 270, 1: 90, 2: 0, 3: 50, 4: 130, 5: 105,
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
                self.kit.servo[ch].set_pulse_width_range(servo.min_pulse, servo.max_pulse)
                self.kit.servo[ch].actuation_range = servo.max_angle
                
                home_angle = self.HOME_POSITION.get(ch, 90)
                self.kit.servo[ch].angle = home_angle
                self.current_angles[ch] = float(home_angle)
                
            self._initialized = True
            print("Initialization Complete. Arm is ready.\n")
        except Exception as e:
            logger.error(f"Servo setup failed: {e}")
            raise

    def move_smooth(self, channel: int, target_angle: float, delay: float = 0.01, step_size: float = 1.0) -> bool:
        """Moves a single servo smoothly to a target angle."""
        if not self._initialized or channel not in self.SERVO_CONFIG:
            logger.error(f"Cannot move channel {channel}: Arm uninitialized or invalid channel.")
            return False

        max_angle = self.SERVO_CONFIG[channel].max_angle
        safe_target = max(0.0, min(float(target_angle), float(max_angle)))

        if safe_target != target_angle:
            logger.warning(f"Ch{channel}: Target {target_angle}° clamped to {safe_target}°")

        current = self.current_angles.get(channel, float(self.HOME_POSITION.get(channel, 90)))
        
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

    def move_all_smooth(self, target_positions: dict[int, int], delay: float = 0.01, max_step: float = 1.0) -> None:
        """Simultaneously moves multiple servos to their target angles proportionally."""
        if not self._initialized:
            logger.error("Cannot move: Arm uninitialized.")
            return

        distances = {}
        start_angles = {}
        
        # 1. Calculate the required travel distance for each valid servo
        for ch, target in target_positions.items():
            if ch not in self.SERVO_CONFIG:
                continue
                
            max_angle = self.SERVO_CONFIG[ch].max_angle
            safe_target = max(0.0, min(float(target), float(max_angle)))
            
            if safe_target != target:
                logger.warning(f"Ch{ch}: Target {target}° clamped to {safe_target}°")
                
            current = self.current_angles.get(ch, float(self.HOME_POSITION.get(ch, 90)))
            start_angles[ch] = current
            distances[ch] = safe_target - current

        if not distances:
            return

        # 2. Find the maximum distance any single servo has to travel
        max_dist = max((abs(d) for d in distances.values()), default=0)
        
        if max_dist < 0.1:
            return # All servos are already at their target positions

        # 3. Determine the number of steps based on the largest movement
        steps = int(max_dist / max_step)
        if steps == 0:
            steps = 1

        # 4. Move all servos incrementally
        for step in range(1, steps + 1):
            for ch in distances.keys():
                # Linear interpolation: move proportionally based on the current step
                current_target = start_angles[ch] + (distances[ch] * (step / steps))
                self.kit.servo[ch].angle = current_target
                self.current_angles[ch] = current_target
            
            # Pause once per step to control overall speed
            time.sleep(delay)
            
        # 5. Lock in the exact final target angles to account for float math drift
        for ch in distances.keys():
            safe_target = start_angles[ch] + distances[ch]
            self.kit.servo[ch].angle = safe_target
            self.current_angles[ch] = safe_target

    def go_home_smooth(self, delay: float = 0.01, max_step: float = 1.0) -> None:
        """Smoothly and simultaneously moves all servos to HOME."""
        print("\nReturning to HOME position smoothly...")
        self.move_all_smooth(self.HOME_POSITION, delay, max_step)
        print("Arm is now at HOME.")

    def go_grab_smooth(self, delay: float = 0.01, max_step: float = 1.0) -> None:
        """Smoothly and simultaneously moves all servos to GRAB position."""
        print("\nMoving to GRAB position smoothly...")
        self.move_all_smooth(self.GRAB_POSITION, delay, max_step)
        print("Arm is now in GRAB position.")
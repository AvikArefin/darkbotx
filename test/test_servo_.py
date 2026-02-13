import time
from adafruit_servokit import ServoKit

def initialize_bot():
    """
    Initializes the 6-servo setup with a red warning countdown.
    Channel 0: 270 degrees, set to 0.
    Channels 1-5: 180 degrees, set to 90.
    """
    # ANSI Escape Codes for colors
    RED   = "\033[91m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    print(f"\n{RED}{BOLD}WARNING: Bot might move abruptly to preset position!!!{RESET}")
    
    # 3... 2... 1... Countdown
    for i in range(3, 0, -1):
        print(f"{RED}{i}...{RESET}")
        time.sleep(1)
    
    print(f"{BOLD}ACTION!{RESET}\n")

    # Initialize the driver (Standard 16-channel PCA9685)
    kit = ServoKit(channels=16)

    # 1. Configure the Gripper (Channel 0)
    # ######################## Range: 270, Position: 270  ########################
    kit.servo[0].set_pulse_width_range(500, 2500)
    kit.servo[0].actuation_range = 270
    kit.servo[0].angle = 270
    print("Channel 0 (Gripper): 270° range set to 270°")
    print("Now the fingers can be attached")

    # 2. Configure the Arm Joints (Channels 1 through 5)
    # ########################  Range: 180, Position: 90  ########################
    for i in range(1, 6):
        kit.servo[i].set_pulse_width_range(500, 2500)
        kit.servo[i].actuation_range = 180
        kit.servo[i].angle = 90
        print(f"Channel {i}: 180° range set to 90°")

    print("\nInitialization Complete. Bot is ready.")
    return kit

if __name__ == "__main__":
    initialize_bot()

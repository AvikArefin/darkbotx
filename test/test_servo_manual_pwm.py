import board
import busio
from adafruit_pca9685 import PCA9685

# Initialize the I2C bus and the PCA9685 controller
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)

# Standard servo frequency is 50Hz
pca.frequency = 50

def send_pulse(channel: int, pulse_us: int):
    """Converts a microsecond pulse into a 16-bit duty cycle and sends it."""
    # 50Hz -> 20,000 us period. Max 16-bit integer is 65535.
    duty_cycle = int((pulse_us / 20000.0) * 65535)
    pca.channels[channel].duty_cycle = duty_cycle

def main():
    target_channel = 0
    
    print("\n--- Direct PWM Tester ---")
    print(f"Hardware reports running at: {pca.frequency} Hz")
    print(f"Testing Channel: {target_channel}")
    print("Warning: Start with safe values (e.g., 1500) to avoid grinding gears.")
    print("Enter 'q' to quit and disable the motor.\n")

    while True:
        user_input = input("Enter pulse width in µs (e.g., 500, 1500, 2500): ")
        
        if user_input.lower() == 'q':
            # Setting duty_cycle to 0 stops sending the signal, relaxing the motor
            pca.channels[target_channel].duty_cycle = 0
            print("Motor relaxed. Exiting.")
            break
            
        try:
            pulse = int(user_input)
            
            # Basic safety clamping to prevent catastrophic hardware damage
            if pulse < 300 or pulse > 2800:
                print("Error: Value dangerously out of bounds. Keep between 300 and 2800.")
                continue
                
            send_pulse(target_channel, pulse)
            print(f"--> Sent {pulse}µs")
            
        except ValueError:
            print("Invalid input. Please enter an integer.")

if __name__ == "__main__":
    main()
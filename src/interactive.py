import sys
import threading
import time

from robot import RobotArm, RJoint
from sensor import SensorMonitor


def sensor_monitor_loop(arm: RobotArm, stop_event: threading.Event):
    monitor = SensorMonitor(arm.sensor)
    while not stop_event.is_set():
        try:
            monitor.update_and_print(overlay=True)
        except Exception:
            pass
        time.sleep(0.1)



def main():
    print("=================== DARKBOTX INTERACTIVE ===================")
    arm = RobotArm()
    stop_event = threading.Event()

    # Clear screen and make room for live header (13 lines)
    sys.stdout.write("\033[2J\033[14;1H")
    sys.stdout.flush()

    monitor_thread = threading.Thread(
        target=sensor_monitor_loop, args=(arm, stop_event), daemon=True
    )
    monitor_thread.start()

    print("Presets: 'home', 'lift', 'put', 'grab'")
    print("Manual Servo: '<channel> <angle>' (e.g. '1 90') | 'q' to quit\n")

    try:
        while True:
            cmd = input("Cmd > ").strip()
            if not cmd:
                continue

            if cmd.lower() in ("q", "quit", "exit"):
                break

            if cmd.lower() == "home":
                arm.go_home_smooth()
            elif cmd.lower() == "lift":
                arm.go_lift_smooth()
            elif cmd.lower() == "put":
                arm.go_put_smooth()
            elif cmd.lower() == "grab":
                arm.go_grab_smooth()
            else:
                parts = cmd.split()
                if len(parts) == 2:
                    try:
                        ch = int(parts[0])
                        angle = float(parts[1])
                        arm.move_smooth(channel=ch, target_angle=angle)
                    except ValueError:
                        print(f"Invalid numbers: '{cmd}'. Format: '<channel> <angle>'")
                else:
                    print(
                        f"Unknown command: '{cmd}'. Use '<channel> <angle>' or 'home'/'lift'/'put'/'grab'/'q'"
                    )

    except KeyboardInterrupt:
        print("\nExiting interactive mode...")
    finally:
        stop_event.set()
        monitor_thread.join(timeout=1.0)
        arm.deinit()
        print("Robot deinitialized cleanly.")


if __name__ == "__main__":
    main()

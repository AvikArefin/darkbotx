# GUI: PyQt imports
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QFrame
from PyQt6.QtCore import Qt, QTimer

# matplotlib imports
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
matplotlib.use('QtAgg')

# custom imports
from code.logger_setup import setup_logger

# CONSTANTS
WIDTH = 800 
HEIGHT = 600

JOINT_NAMES = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']
JOINT_LOWER_LIMITS = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671, 0., 0.]
JOINT_UPPER_LIMITS = [2.9671, 1.8326, 2.9671, 0. , 2.9671, 3.8223, 2.9671, 0.04, 0.04] 

logger = setup_logger(__name__)

class FloatSlider(QWidget):
    """
    A wrapper around QSlider that displays the current float value in its label.
    """
    def __init__(self, min_val, max_val, label_text, parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.range_span = max_val - min_val
        self.name = label_text  # Store the joint name
        
        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Label (Initialize with 0.000 or min_val)
        self.label = QLabel(f"{self.name}: {0.0:.4f}")
        layout.addWidget(self.label)
        
        # Slider (Mapped to 0-1000 for resolution)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setEnabled(False) # Disable manual control since sim drives it
        layout.addWidget(self.slider)

    def set(self, float_value):
        """Sets the slider position and updates the label with the value."""
        # Clamp value
        val = max(self.min_val, min(float_value, self.max_val))
        
        # Map float to 0-1000 range
        normalized = (val - self.min_val) / self.range_span if self.range_span != 0 else 0
        self.slider.setValue(int(normalized * 1000))
        
        # Update the text label with the actual float value
        self.label.setText(f"{self.name}: {self.min_val} < {val:.4f} < {self.max_val}")

class RewardPlotter(FigureCanvas):
    def __init__(self, parent=None):
        plt.rcParams.update({
            'font.family': 'Courier New',  # monospace font
            'font.size': 10,
            'axes.titlesize': 10,
            'axes.labelsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 10
        })
        self.figure = Figure(figsize=(6, 4), dpi=100)

        # ax configs
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Episode Reward Trace")

        self.ax.grid(True, which="major", linestyle="-", linewidth=1.0, alpha=0.70)
        self.ax.minorticks_on()
        self.ax.grid(True, which="minor", linestyle="-", linewidth=1.0, alpha=0.50)
        self.ax.set_axisbelow(True)

        self.rewards = []
        self.line, = self.ax.plot([], [], '-', linewidth=2.0, alpha=1.0)
        
        super().__init__(self.figure)
        if parent:
            parent.layout().addWidget(self)

    def update_plot(self, new_reward):
        self.rewards.append(new_reward)
        self.line.set_data(range(len(self.rewards)), self.rewards)
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.grid(True, which="major", linestyle="-", linewidth=1.0, alpha=0.30)
        self.ax.minorticks_on()
        self.ax.grid(True, which="minor", linestyle="-", linewidth=1.0, alpha=0.15)
        self.ax.set_axisbelow(True)
        self.draw()

    def reset_plot(self):
        """Clears data for a new episode."""
        if self.rewards:
            self.line.set_alpha(0.5)
            self.line.set_linewidth(1.0)

        self.rewards = [0] # Start at 0

        self.line, = self.ax.plot([], [], '-', linewidth=2.0, alpha=1.0)
        self.draw()

class Monitor:
    """Encapsulates the GUI monitoring window and its components."""
    
    def __init__(self, argv: list[str]) -> None:
        """Initialize the Monitor with Qt application and main window."""
        self.app: QApplication = QApplication(argv)
        self.window = QMainWindow()
        central_widget = QWidget()
        self.window.setCentralWidget(central_widget)
        self.joints_var: list[FloatSlider] = []
        self.timer: QTimer | None = None
        
        self.setup_ui(central_widget)
        self.window.show()
        self.update_gui()
    
    def setup_ui(self, central_widget: QWidget) -> None:
        """Initializes all the UI elements."""
        self.window.setWindowTitle("Monitor")
        self.window.resize(WIDTH, HEIGHT)
        
        # Connect close event
        self.window.closeEvent = self.on_close
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header_label = QLabel("MONITOR")
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        main_layout.addWidget(header_label)
        
        # Episode label
        self.episode_label = QLabel("Episode: -")
        self.episode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.episode_label)
        
        # Content layout (side-by-side: joints and plot)
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)
        
        # --- Left: Joint sliders ---
        joint_frame = QWidget()
        joint_layout = QVBoxLayout(joint_frame)
        content_layout.addWidget(joint_frame, stretch=1)
        
        # Create joint sliders
        for name, lower, upper in zip(JOINT_NAMES, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS):
            slider_widget = FloatSlider(lower, upper, name)
            joint_layout.addWidget(slider_widget)
            self.joints_var.append(slider_widget)
        
        # Cube info label
        self.cube_label = QLabel("Cube: -")
        self.cube_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        joint_layout.addWidget(self.cube_label)
        joint_layout.addStretch()
        
        # --- Right: Reward plot ---
        reward_plot_frame = QWidget()
        reward_layout = QVBoxLayout(reward_plot_frame)
        content_layout.addWidget(reward_plot_frame, stretch=2)
        
        self.plotter = RewardPlotter(reward_plot_frame)
    
    def update_gui(self) -> None:
        """Processes GUI events to keep the window responsive."""
        self.app.processEvents()
    
    def set_episode(self, episode: int) -> None:
        """Updates the episode label."""
        self.episode_label.setText(f"Episode: {episode}")
    
    def set_cube_label(self, cube_info: str) -> None:
        """Updates the cube information label."""
        self.cube_label.setText(f"Cube: {cube_info}")
    
    def update_plot(self, reward: float) -> None:
        """Adds a new reward value to the plot."""
        self.plotter.update_plot(reward)
    
    def reset_plot(self) -> None:
        """Resets the reward plot for a new episode."""
        self.plotter.reset_plot()
    
    def update_joints(self, info: dict) -> None:
        """Updates all joint sliders from simulation info."""
        target_positions = info.get("target_joints_pos", [])
        for slider_widget, value in zip(self.joints_var, target_positions):
            slider_widget.set(value)
    
    def start_manual_loop(self, step_callback) -> None:
        """
        Starts a 60FPS QTimer to drive the simulation step_callback.
        Enables all sliders for manual control.
        
        Args:
            step_callback: Function to call on each timer tick
        """
        # Enable all sliders for manual control
        for slider_widget in self.joints_var:
            slider_widget.slider.setEnabled(True)
        
        # Create and start timer
        self.timer = QTimer()
        self.timer.timeout.connect(step_callback)
        self.timer.start(16)  # ~60 FPS (16ms)
    
    def stop_manual_loop(self) -> None:
        """Stops the manual control timer if running."""
        if self.timer is not None and self.timer.isActive():
            self.timer.stop()
    
    def on_close(self, event) -> None:
        """Handle window close event - cleanup resources."""
        logger.info("Monitor window closing...")
        self.stop_manual_loop()
        event.accept()  # Accept the close event
    
    def exec(self) -> int:
        """
        Start the Qt event loop and block until window is closed.
        
        Returns:
            Exit code from Qt application
        """
        return self.app.exec()
    
    def close(self) -> None:
        """Clean up resources and close the window."""
        self.stop_manual_loop()
        self.window.close()


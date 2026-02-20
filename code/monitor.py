import math

# GUI: PyQt imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QSlider, QFrame, QGridLayout, QScrollArea
)
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
WIDTH = 1200 
HEIGHT = 800

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
        self.name = label_text  
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(f"{self.name}: {0.0:.4f}")
        layout.addWidget(self.label)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setEnabled(False) 
        layout.addWidget(self.slider)

    def set(self, float_value):
        """Sets the slider position and updates the label with the value."""
        val = max(self.min_val, min(float_value, self.max_val))
        normalized = (val - self.min_val) / self.range_span if self.range_span != 0 else 0
        self.slider.setValue(int(normalized * 1000))
        self.label.setText(f"{self.name}: {self.min_val} < {val:.4f} < {self.max_val}")

class RewardPlotter(FigureCanvas):
    def __init__(self, parent=None):
        plt.rcParams.update({
            'font.family': 'Courier New',  
            'font.size': 8,
            'axes.titlesize': 10,
            'axes.labelsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 10
        })
        # Reduced size so multiple plots fit well in the grid
        self.figure = Figure(figsize=(4, 3), dpi=100)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Reward Trace")

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
        self.rewards = [0]
        self.line, = self.ax.plot([], [], '-', linewidth=2.0, alpha=1.0)
        self.draw()

class EnvPanel(QFrame):
    """A standalone panel containing sliders and a plot for a single environment."""
    def __init__(self, env_idx: int, parent=None):
        super().__init__(parent)
        self.env_idx = env_idx
        self.joints_var: list[FloatSlider] = []
        
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setLineWidth(2)
        
        layout = QVBoxLayout(self)
        
        title = QLabel(f"Environment {self.env_idx}")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #333; color: white; padding: 4px;")
        layout.addWidget(title)
        
        content_layout = QHBoxLayout()
        layout.addLayout(content_layout)
        
        # --- Left: Joint sliders ---
        joint_layout = QVBoxLayout()
        content_layout.addLayout(joint_layout, stretch=1)
        
        for name, lower, upper in zip(JOINT_NAMES, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS):
            slider_widget = FloatSlider(lower, upper, name)
            joint_layout.addWidget(slider_widget)
            self.joints_var.append(slider_widget)
            
        self.cube_label = QLabel("Cube: -")
        self.cube_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        joint_layout.addWidget(self.cube_label)
        joint_layout.addStretch()
        
        # --- Right: Reward plot ---
        reward_plot_frame = QWidget()
        reward_layout = QVBoxLayout(reward_plot_frame)
        reward_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(reward_plot_frame, stretch=2)
        
        self.plotter = RewardPlotter(reward_plot_frame)

    def set_cube_label(self, cube_info: str) -> None:
        self.cube_label.setText(f"Cube: {cube_info}")
        
    def update_plot(self, reward: float) -> None:
        self.plotter.update_plot(reward)
        
    def reset_plot(self) -> None:
        self.plotter.reset_plot()

    def update_joints(self, target_positions: list) -> None:
        for slider_widget, value in zip(self.joints_var, target_positions):
            slider_widget.set(value)

class Monitor:
    """Encapsulates the GUI monitoring window with a dynamic grid of environments."""
    
    def __init__(self, argv: list[str], num_envs: int = 1) -> None:
        self.app: QApplication = QApplication(argv)
        self.window = QMainWindow()
        self.num_envs = num_envs
        
        self.env_panels: list[EnvPanel] = []
        self.timer: QTimer | None = None
        
        self.setup_ui()
        self.window.show()
        self.update_gui()
    
    def setup_ui(self) -> None:
        self.window.setWindowTitle(f"Monitor - {self.num_envs} Environments")
        self.window.resize(WIDTH, HEIGHT)
        self.window.closeEvent = self.on_close
        
        # Main widget
        main_widget = QWidget()
        self.window.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Header
        header_layout = QHBoxLayout()
        title_label = QLabel("MULTI-ENVIRONMENT MONITOR")
        title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        
        self.episode_label = QLabel("Episode: -")
        self.episode_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.episode_label)
        main_layout.addLayout(header_layout)
        
        # --- Scrollable Grid Area ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)
        
        scroll_content = QWidget()
        scroll_area.setWidget(scroll_content)
        
        self.grid_layout = QGridLayout(scroll_content)
        
        # Calculate row/col count
        n_cols = math.ceil(math.sqrt(self.num_envs))
        
        # Instantiate an EnvPanel for each environment
        for i in range(self.num_envs):
            row = i // n_cols
            col = i % n_cols
            
            panel = EnvPanel(env_idx=i)
            self.grid_layout.addWidget(panel, row, col)
            self.env_panels.append(panel)
    
    def update_gui(self) -> None:
        self.app.processEvents()
    
    def set_episode(self, episode: int) -> None:
        self.episode_label.setText(f"Episode: {episode}")
    
    def start_manual_loop(self, step_callback) -> None:
        """Enables sliders for all environments and starts physics timer."""
        for panel in self.env_panels:
            for slider_widget in panel.joints_var:
                slider_widget.slider.setEnabled(True)
                
        self.timer = QTimer()
        self.timer.timeout.connect(step_callback)
        self.timer.start(16)  # ~60 FPS
    
    def stop_manual_loop(self) -> None:
        if self.timer is not None and self.timer.isActive():
            self.timer.stop()
    
    def on_close(self, event) -> None:
        logger.info("Monitor window closing...")
        self.stop_manual_loop()
        event.accept() 
    
    def exec(self) -> int:
        return self.app.exec()
    
    def close(self) -> None:
        self.stop_manual_loop()
        self.window.close()

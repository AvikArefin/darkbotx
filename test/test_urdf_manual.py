import numpy as np
import genesis as gs
import pygame
import tkinter as tk
from tkinter import simpledialog
from enum import Enum, auto

WINDOW_SIZE = (420, 800)
FPS = 60


class Joint(Enum):
    """Logical names for the robot joints."""

    GRIPPER = auto()
    WRIST_ROT = auto()
    WRIST = auto()
    ELBOW = auto()
    SHOULDER = auto()
    BASE = auto()


# Centralized mapping: logical joint -> URDF joint name
URDF_MAPPING: dict[Joint, str] = {
    Joint.GRIPPER: "Slider 37",
    Joint.WRIST_ROT: "Revolute 19",
    Joint.WRIST: "Revolute 35",
    Joint.ELBOW: "Revolute 34",
    Joint.SHOULDER: "Revolute 36",
    Joint.BASE: "Revolute 13",
}

# Create a reverse mapping for efficient lookup during initialization
REVERSE_URDF_MAPPING = {v: k for k, v in URDF_MAPPING.items()}


def ask_value(name, current_display, unit, min_display, max_display):
    """
    A native OS modal dialog is opened to obtain a numeric value from the user.
    Keyboard focus is maintained by blocking the main thread execution.
    """
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)
    result = simpledialog.askfloat(
        title=f"Set {name}",
        prompt=f"{name}  ({unit})\nRange: [{min_display:.1f}, {max_display:.1f}]",
        initialvalue=round(current_display, 2),
        minvalue=min_display,
        maxvalue=max_display,
    )
    root.destroy()
    return result


class Slider:
    def __init__(
        self,
        x,
        y,
        w,
        h,
        min_val,
        max_val,
        raw_name,
        logical_joint=None,
        is_rotational=False,
        initial_value=None,
    ):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.raw_name = raw_name
        self.logical_joint = logical_joint

        # The display name is constructed using the Enum name and the raw URDF string.
        if logical_joint:
            self.name = f"{logical_joint.name} ({raw_name})"
        else:
            self.name = raw_name

        self.is_rotational = is_rotational
        self.value = (
            initial_value if initial_value is not None else (min_val + max_val) / 2
        )
        self.dragging = False
        self.last_mouse_x = 0
        self.button_w = 10
        self.edit_btn_rect = pygame.Rect(x + w + 5, y, 60, h)

    def to_display_value(self, val):
        return np.rad2deg(val) if self.is_rotational else val

    def from_display_value(self, val):
        return np.deg2rad(val) if self.is_rotational else val

    @property
    def unit(self):
        return "deg" if self.is_rotational else "m"

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.edit_btn_rect.collidepoint(event.pos):
                return True
            elif self.rect.collidepoint(event.pos):
                self.dragging = True
                self.last_mouse_x = event.pos[0]
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            delta_x = event.pos[0] - self.last_mouse_x
            self.last_mouse_x = event.pos[0]
            range_val = self.max_val - self.min_val
            ppu = self.rect.w / range_val if range_val != 0 else 1
            self.value = max(
                self.min_val, min(self.max_val, self.value + delta_x / ppu)
            )
        return False

    def open_dialog(self):
        dv = self.to_display_value(self.value)
        min_d = self.to_display_value(self.min_val)
        max_d = self.to_display_value(self.max_val)
        result = ask_value(self.name, dv, self.unit, min_d, max_d)
        if result is not None:
            self.value = max(
                self.min_val, min(self.max_val, self.from_display_value(result))
            )

    def draw(self, surface, font):
        pygame.draw.rect(surface, (128, 128, 128), self.rect)
        t = (
            (self.value - self.min_val) / (self.max_val - self.min_val)
            if self.max_val != self.min_val
            else 0
        )
        thumb_x = self.rect.x + int(t * (self.rect.w - self.button_w))
        pygame.draw.rect(
            surface,
            (255, 255, 255),
            pygame.Rect(thumb_x, self.rect.y, self.button_w, self.rect.h),
        )
        dv = self.to_display_value(self.value)
        label = font.render(f"{self.name}: {dv:.1f}{self.unit}", True, (0, 0, 0))
        surface.blit(label, (self.rect.x, self.rect.y - 20))
        pygame.draw.rect(surface, (70, 130, 180), self.edit_btn_rect)
        btn_label = font.render("edit", True, (255, 255, 255))
        bx = self.edit_btn_rect.x + (self.edit_btn_rect.w - btn_label.get_width()) // 2
        by = self.edit_btn_rect.y + (self.edit_btn_rect.h - btn_label.get_height()) // 2
        surface.blit(btn_label, (bx, by))


def main():
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=True,
        sim_options=gs.options.SimOptions(dt=0.002, gravity=(0, 0, -10.0)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 1.5), camera_lookat=(0.0, 0.0, 0.5), camera_fov=30
        ),
        show_FPS=False,
    )

    _ = scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="assets/Hiwonder_urdf_simplified/Hiwonder.urdf",
            fixed=True,
            decimate=False,
            convexify=False,
        ),
        # vis_mode="collision",
    )

    box = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.0, -0.1, 0.4)),
        material=gs.materials.Rigid(friction=2.5, rho=500.0),
        vis_mode="collision",
    )

    scene.build()

    robot.set_dofs_kp(np.ones(robot.n_dofs) * 3000.0)
    robot.set_dofs_kv(np.ones(robot.n_dofs) * 300.0)
    robot.set_dofs_force_range(
        np.ones(robot.n_dofs) * -1000.0, np.ones(robot.n_dofs) * 1000.0
    )

    # Default values are defined using the Joint Enum keys.
    default_vals = {
        Joint.BASE: np.deg2rad(0.0),
        Joint.SHOULDER: np.deg2rad(122.0),
        Joint.ELBOW: np.deg2rad(51.4),
        Joint.WRIST: np.deg2rad(90.0),
        Joint.WRIST_ROT: np.deg2rad(90.6),
        Joint.GRIPPER: -0.0,
    }

    controlled_dofs = []
    for joint in robot.joints:
        pos_limit = joint.dofs_limit
        if pos_limit[0] is not None:
            logical_key = REVERSE_URDF_MAPPING.get(joint.name)
            if logical_key:
                controlled_dofs.append(
                    {
                        "raw_name": joint.name,
                        "logical_joint": logical_key,
                        "idx": joint.dofs_idx_local[0],
                        "min": pos_limit[0][0],
                        "max": pos_limit[0][1],
                        "is_rotational": joint.type == gs.JOINT_TYPE.REVOLUTE,
                    }
                )

    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Joint Control")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    sliders = []
    for i, d in enumerate(controlled_dofs):
        sliders.append(
            Slider(
                20,
                60 + i * 60,
                280,
                30,
                d["min"],
                d["max"],
                d["raw_name"],
                logical_joint=d["logical_joint"],
                is_rotational=d["is_rotational"],
                initial_value=default_vals.get(
                    d["logical_joint"], (d["min"] + d["max"]) / 2
                ),
            )
        )

    box_pos_slider_x = Slider(
        20,
        60 + len(controlled_dofs) * 60,
        280,
        30,
        -2.0,
        2.0,
        "Box X",
        initial_value=0.0,
    )
    box_pos_slider_y = Slider(
        20,
        60 + (len(controlled_dofs) + 1) * 60,
        280,
        30,
        -2.0,
        2.0,
        "Box Y",
        initial_value=-0.1,
    )
    sliders.extend([box_pos_slider_x, box_pos_slider_y])

    print_btn_rect = pygame.Rect(20, 60 + (len(controlled_dofs) + 2) * 60, 100, 30)
    target_positions = np.zeros(robot.n_dofs)

    running = True
    while running:
        dialog_slider = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and print_btn_rect.collidepoint(
                event.pos
            ):
                for s in sliders:
                    print(f"{s.name}: {s.to_display_value(s.value):.3f}{s.unit}")
            for slider in sliders:
                if slider.handle_event(event):
                    dialog_slider = slider

        if dialog_slider:
            dialog_slider.open_dialog()

        for i, slider in enumerate(sliders[: len(controlled_dofs)]):
            target_positions[controlled_dofs[i]["idx"]] = slider.value

        current_box_pos = box.get_pos()
        if hasattr(current_box_pos, "cpu"):
            current_box_pos = current_box_pos.cpu().numpy()

        if box_pos_slider_x.dragging or box_pos_slider_y.dragging:
            box.set_pos(
                [box_pos_slider_x.value, box_pos_slider_y.value, current_box_pos[2]]
            )
        else:
            box_pos_slider_x.value, box_pos_slider_y.value = (
                current_box_pos[0],
                current_box_pos[1],
            )

        robot.control_dofs_position(target_positions)
        scene.step()

        screen.fill((255, 255, 255))
        for slider in sliders:
            slider.draw(screen, font)
        pygame.draw.rect(screen, (100, 180, 100), print_btn_rect)
        screen.blit(
            font.render("Print All", True, (255, 255, 255)),
            (print_btn_rect.x + 10, print_btn_rect.y + 5),
        )
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()

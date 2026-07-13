from config import TEST_ENV_CFG
import numpy as np
import genesis as gs
import pygame
import tkinter as tk
from tkinter import simpledialog
import torch
import sys
import os
from typing import TypedDict

# Ensure we can import from the current directory (src)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import GraspEnv
from config import SJoint, TEST_ENV_CFG

class DofDict(TypedDict):
    raw_name: str
    logical_joint: SJoint
    idx: int
    min: float
    max: float
    is_rotational: bool
    initial_value: float

WINDOW_SIZE = (420, 850)
FPS = 60

def ask_value(name: str, current_display: float, unit: str, min_display: float, max_display: float) -> float | None:
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
        x: int,
        y: int,
        w: int,
        h: int,
        min_val: float,
        max_val: float,
        raw_name: str,
        logical_joint: SJoint | None = None,
        is_rotational: bool = False,
        initial_value: float | None = None,
    ) -> None:
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

    def to_display_value(self, val: float) -> float:
        return float(np.rad2deg(val)) if self.is_rotational else float(val)

    def from_display_value(self, val: float) -> float:
        return float(np.deg2rad(val)) if self.is_rotational else float(val)

    @property
    def unit(self) -> str:
        return "deg" if self.is_rotational else "m"

    def handle_event(self, event: pygame.event.Event) -> bool:
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

    def open_dialog(self) -> None:
        dv = self.to_display_value(self.value)
        min_d = self.to_display_value(self.min_val)
        max_d = self.to_display_value(self.max_val)
        result = ask_value(self.name, dv, self.unit, min_d, max_d)
        if result is not None:
            self.value = max(
                self.min_val, min(self.max_val, self.from_display_value(result))
            )

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
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


def main() -> None:
    gs.init(backend=gs.gpu) # pyrefly: ignore[missing-attribute]
    
    # Initialize the 1-to-1 environment
    env = GraspEnv(TEST_ENV_CFG)
    robot = env.robot._robot_entity
    
    # Optional: explicitly call env.reset() to put everything in the initial state
    env.reset()

    controlled_dofs: list[DofDict] = []
    # get_qpos() returns [num_envs, n_dofs], take the first env
    initial_qpos = robot.get_qpos()[0].cpu().numpy()

    # Create UI descriptors from HiwonderJoint enum, mapping directly to Manipulator's limits
    for joint_enum in SJoint:
        if joint_enum.name == "GRIPPER_RIGHT":
            continue
        idx = joint_enum.value
        
        # We wrap in try-except because GraspEnv lazily builds limit cache
        try:
            limit = env.robot._get_limit(idx)
        except KeyError:
            # If the joint isn't active/controlled, we skip
            continue
            
        is_rotational = True
        raw_name = joint_enum.name
        
        # Retrieve original name and type from Genesis joint structures
        for joint in robot.joints:
            if len(joint.dofs_idx_local) > 0 and joint.dofs_idx_local[0] == idx:
                is_rotational = (joint.type == gs.JOINT_TYPE.REVOLUTE)
                raw_name = joint.name
                break

        controlled_dofs.append({
            "raw_name": raw_name,
            "logical_joint": joint_enum,
            "idx": idx,
            "min": limit[0],
            "max": limit[1],
            "is_rotational": is_rotational,
            "initial_value": initial_qpos[idx]
        })

    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("DarkBotX - Environment Manual UI")
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
                initial_value=d["initial_value"],
            )
        )

    # Add object position sliders
    current_object_pos = env.objects.get_pos()[0].cpu().numpy()

    box_pos_slider_x = Slider(
        20,
        60 + len(controlled_dofs) * 60,
        280,
        30,
        -0.5,
        0.5,
        "Object X",
        initial_value=current_object_pos[0],
    )
    box_pos_slider_y = Slider(
        20,
        60 + (len(controlled_dofs) + 1) * 60,
        280,
        30,
        -0.5,
        0.5,
        "Object Y",
        initial_value=current_object_pos[1],
    )
    sliders.extend([box_pos_slider_x, box_pos_slider_y])

    print_btn_rect = pygame.Rect(20, 60 + (len(controlled_dofs) + 2) * 60, 100, 30)
    btn_grasp_rect = pygame.Rect(130, 60 + (len(controlled_dofs) + 2) * 60, 110, 30)
    btn_lift_rect = pygame.Rect(250, 60 + (len(controlled_dofs) + 2) * 60, 110, 30)
    btn_reset_rect = pygame.Rect(20, 60 + (len(controlled_dofs) + 3) * 60, 100, 30)

    running = True
    auto_move_frames = 0
    step_counter = 0
    while running:
        step_counter += 1
        if step_counter % 20 == 0:
            env.get_observations()
        dialog_slider = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if print_btn_rect.collidepoint(event.pos):
                    for s in sliders:
                        print(f"{s.name} [Target]: {s.to_display_value(s.value):.3f}{s.unit}")
                    actual_qpos = env.robot._robot_entity.get_qpos()[0].cpu().numpy()
                    print(f"\n--- Physical State ---")
                    print(f"Actual Left Gripper: {actual_qpos[SJoint.GRIPPER_LEFT]:.5f} m")
                    print(f"----------------------\n")
                    sys.stdout.flush()
                elif btn_grasp_rect.collidepoint(event.pos):
                    env.robot.move_to_grasp_position()
                    auto_move_frames = 120
                elif btn_lift_rect.collidepoint(event.pos):
                    env.robot.move_to_lift_position()
                    auto_move_frames = 120
                elif btn_reset_rect.collidepoint(event.pos):
                    env.reset()
                    auto_move_frames = 10
            for slider in sliders:
                if slider.handle_event(event):
                    dialog_slider = slider

        if dialog_slider:
            dialog_slider.open_dialog()

        if auto_move_frames > 0:
            auto_move_frames -= 1
            current_qpos = env.robot._robot_entity.get_qpos()[0].cpu().numpy()
            for i, slider in enumerate(sliders[: len(controlled_dofs)]):
                slider.value = current_qpos[controlled_dofs[i]["idx"]]
        else:
            # Update targets based on sliders
            targets: dict[SJoint, int | float | torch.Tensor] = {}
            for i, slider in enumerate(sliders[: len(controlled_dofs)]):
                logical = controlled_dofs[i]["logical_joint"]
                if isinstance(logical, SJoint):
                    targets[logical] = float(slider.value)
    
            # Use Manipulator to apply targets
            env.robot.control_motors(targets)

        # Handle object position
        current_object_pos = env.objects.get_pos()[0].cpu().numpy()
        if box_pos_slider_x.dragging or box_pos_slider_y.dragging:
            # Sync slider state to physics state
            new_pos = torch.tensor(
                [[box_pos_slider_x.value, box_pos_slider_y.value, current_object_pos[2]]],
                device=gs.device,
                dtype=torch.float32
            )
            env.objects.set_pos(new_pos, envs_idx=torch.tensor([0], device=gs.device))
        else:
            # Sync physics state to slider state (in case something bumps it)
            box_pos_slider_x.value, box_pos_slider_y.value = (
                current_object_pos[0],
                current_object_pos[1],
            )

        # Step the scene manually (bypassing the env.step() RL trajectory sequence)
        try:
            env.scene.step()
        except gs.GenesisException as e:
            # Exits cleanly if the Genesis 3D viewer is closed by the user
            print(f"Genesis Exception: {e}")
            running = False
            break

        screen.fill((255, 255, 255))
        for slider in sliders:
            slider.draw(screen, font)
            
        pygame.draw.rect(screen, (100, 180, 100), print_btn_rect)
        screen.blit(
            font.render("Print", True, (255, 255, 255)),
            (print_btn_rect.x + 25, print_btn_rect.y + 5),
        )
        
        pygame.draw.rect(screen, (180, 100, 100), btn_grasp_rect)
        screen.blit(
            font.render("Grasp", True, (255, 255, 255)),
            (btn_grasp_rect.x + 30, btn_grasp_rect.y + 5),
        )
        
        pygame.draw.rect(screen, (100, 100, 180), btn_lift_rect)
        screen.blit(
            font.render("Lift", True, (255, 255, 255)),
            (btn_lift_rect.x + 35, btn_lift_rect.y + 5),
        )
        
        pygame.draw.rect(screen, (200, 150, 50), btn_reset_rect)
        screen.blit(
            font.render("Reset", True, (255, 255, 255)),
            (btn_reset_rect.x + 25, btn_reset_rect.y + 5),
        )
        
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()

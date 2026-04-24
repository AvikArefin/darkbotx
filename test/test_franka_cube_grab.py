import numpy as np
import genesis as gs
import pygame
import tkinter as tk
from tkinter import simpledialog

WINDOW_SIZE = (420, 800)
FPS = 60


def ask_value(name, current_display, unit, min_display, max_display):
    """
    Open a native OS modal dialog to get a numeric value from the user.
    Runs on the main thread, blocking everything (including scene.step),
    so GLFW / Genesis cannot steal keyboard focus.
    Returns the new internal value, or None if the user cancelled.
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
        name,
        is_rotational=False,
        initial_value=None,
    ):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.name = name
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
                self.min_val,
                min(self.max_val, self.from_display_value(result)),
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
        label = font.render(f"{self.name}: {dv:.3f}{self.unit}", True, (0, 0, 0))
        surface.blit(label, (self.rect.x, self.rect.y - 20))

        pygame.draw.rect(surface, (70, 130, 180), self.edit_btn_rect)
        pygame.draw.rect(surface, (30, 80, 130), self.edit_btn_rect, 1)
        btn_label = font.render("edit", True, (255, 255, 255))
        bx = self.edit_btn_rect.x + (self.edit_btn_rect.w - btn_label.get_width()) // 2
        by = self.edit_btn_rect.y + (self.edit_btn_rect.h - btn_label.get_height()) // 2
        surface.blit(btn_label, (bx, by))


def main():
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=True,
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0, 0, -10.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
        ),
        show_FPS=False,
    )

    _ = scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        vis_mode='collision',
    )

    box = scene.add_entity(
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.5, 0.0, 0.02)),
    )

    scene.build()

    # --- ADD THIS TO FIX ACTUATOR STRENGTH ---
    # Apply high stiffness (kp) and damping (kv) so the joints actually follow position targets
    kp = np.ones(robot.n_dofs) * 4500.0  
    kv = np.ones(robot.n_dofs) * 450.0   
    robot.set_dofs_kp(kp)
    robot.set_dofs_kv(kv)
    
    # Ensure the internal motors are allowed to apply enough force to move
    max_force = np.ones(robot.n_dofs) * 1000.0
    robot.set_dofs_force_range(-max_force, max_force)
    # -----------------------------------------

    # Grab the true starting state to prevent passive joints from snapping to zero
    init_dof_pos = robot.get_dofs_position()
    if hasattr(init_dof_pos, 'cpu'):
        init_dof_pos = init_dof_pos.cpu().numpy()
    else:
        init_dof_pos = np.array(init_dof_pos)
        
    target_positions = init_dof_pos.copy()

    controlled_dofs = []
    # Automatically map ALL valid revolute and prismatic joints
    for joint in robot.joints:
        pos_limit = joint.dofs_limit
        if pos_limit[0] is not None:
            is_revolute = joint.type == gs.JOINT_TYPE.REVOLUTE
            is_prismatic = joint.type == gs.JOINT_TYPE.PRISMATIC
            
            if is_revolute or is_prismatic:
                controlled_dofs.append(
                    {
                        "name": joint.name,
                        "idx": joint.dofs_idx_local[0],
                        "min": pos_limit[0][0],
                        "max": pos_limit[0][1],
                        "is_rotational": is_revolute,
                    }
                )

    default_vals = {
        "joint1": np.deg2rad(1.186),
        "joint2": np.deg2rad(55.551),
        "joint3": np.deg2rad(0.0),
        "joint4": np.deg2rad(-74.643),
        "joint5": np.deg2rad(0.0),
        "joint6": np.deg2rad(116.229),
        "joint7": np.deg2rad(37.944),
        "finger_joint1": 0.035,
        "finger_joint2": 0.035,
    }
    
    sliders = [
        Slider(
            20,
            60 + i * 50, # Tightened spacing slightly to fit more sliders
            280,
            25,
            d["min"],
            d["max"],
            d["name"],
            d["is_rotational"],
            initial_value=default_vals.get(d["name"], (d["min"] + d["max"]) / 2),
        )
        for i, d in enumerate(controlled_dofs)
    ]
    
    box_pos_slider_x = Slider(
        20,
        60 + len(controlled_dofs) * 50 + 20,
        280,
        25,
        -1.0,
        1.0,
        "Box X",
        initial_value=0.671,
    )
    box_pos_slider_y = Slider(
        20,
        60 + len(controlled_dofs) * 50 + 70,
        280,
        25,
        -1.0,
        1.0,
        "Box Y",
        initial_value=0.207,
    )
    
    sliders.append(box_pos_slider_x)
    sliders.append(box_pos_slider_y)

    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Joint Control")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    print_btn_rect = pygame.Rect(20, 60 + len(controlled_dofs) * 50 + 120, 100, 30)
    print_requested = False

    running = True
    while running:
        dialog_slider = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if print_btn_rect.collidepoint(event.pos):
                    print_requested = True
            for slider in sliders:
                if slider.handle_event(event):
                    dialog_slider = slider

        if dialog_slider is not None:
            dialog_slider.open_dialog()

        # Quality of Life: Synchronize the finger sliders if one is dragged
        finger1_slider = next((s for s in sliders if s.name == "finger_joint1"), None)
        finger2_slider = next((s for s in sliders if s.name == "finger_joint2"), None)
        
        if finger1_slider and finger2_slider:
            if finger1_slider.dragging:
                finger2_slider.value = finger1_slider.value
            elif finger2_slider.dragging:
                finger1_slider.value = finger2_slider.value

        for i, slider in enumerate(sliders[: len(controlled_dofs)]):
            idx = controlled_dofs[i]["idx"]
            target_positions[idx] = slider.value

        if print_requested:
            for i, slider in enumerate(sliders[: len(controlled_dofs)]):
                print(
                    f"{controlled_dofs[i]['name']}: {slider.to_display_value(slider.value):.3f}{slider.unit}"
                )
            for s in sliders[len(controlled_dofs) :]:
                print(f"{s.name}: {s.to_display_value(s.value):.3f}{s.unit}")
            print_requested = False

        # --- TWO-WAY SYNCHRONIZATION LOGIC ---
        current_pos = box.get_pos()
        if hasattr(current_pos, 'cpu'):
            current_pos = current_pos.cpu().numpy()
        else:
            current_pos = np.array(current_pos)
        
        if box_pos_slider_x.dragging or box_pos_slider_y.dragging:
            current_pos[0] = box_pos_slider_x.value
            current_pos[1] = box_pos_slider_y.value
            box.set_pos(current_pos)
        else:
            box_pos_slider_x.value = current_pos[0]
            box_pos_slider_y.value = current_pos[1]
        # -------------------------------------

        robot.control_dofs_position(target_positions)
        scene.step()

        screen.fill((255, 255, 255))
        title = font.render("Joint Position Controls", True, (0, 0, 0))
        screen.blit(title, (20, 20))
        for slider in sliders:
            slider.draw(screen, font)

        pygame.draw.rect(screen, (100, 180, 100), print_btn_rect)
        pygame.draw.rect(screen, (50, 130, 50), print_btn_rect, 1)
        btn_label = font.render("Print All", True, (255, 255, 255))
        bx = print_btn_rect.x + (print_btn_rect.w - btn_label.get_width()) // 2
        by = print_btn_rect.y + (print_btn_rect.h - btn_label.get_height()) // 2
        screen.blit(btn_label, (bx, by))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
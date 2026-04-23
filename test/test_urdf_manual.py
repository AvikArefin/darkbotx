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

        # Clickable edit button to the right of the slider
        self.edit_btn_rect = pygame.Rect(x + w + 5, y, 60, h)

    def to_display_value(self, val):
        return np.rad2deg(val) if self.is_rotational else val

    def from_display_value(self, val):
        return np.deg2rad(val) if self.is_rotational else val

    @property
    def unit(self):
        return "deg" if self.is_rotational else "m"

    def handle_event(self, event):
        """
        Returns True if this slider consumed the event and wants to open
        the value dialog (caller should do so after all sliders have handled).
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.edit_btn_rect.collidepoint(event.pos):
                return True  # signal: open dialog for this slider
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
        """Block, show OS dialog, update value if user confirms."""
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
        # Slider track
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

        # Label
        dv = self.to_display_value(self.value)
        label = font.render(f"{self.name}: {dv:.1f}{self.unit}", True, (0, 0, 0))
        surface.blit(label, (self.rect.x, self.rect.y - 20))

        # Edit button
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
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
        ),
        show_FPS=False,
    )

    plane = scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="assets/Hiwonder_description/Hiwonder.urdf",
            fixed=True,
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(size=(0.015, 0.015, 0.015), pos=(0.0, 0.0, 0.0)),
    )

    scene.build()

    controlled_dofs = []
    for joint in robot.joints:
        pos_limit = joint.dofs_limit
        if pos_limit[0] is not None:
            is_revolute = joint.type == gs.JOINT_TYPE.REVOLUTE
            is_master_slider = (
                joint.type == gs.JOINT_TYPE.PRISMATIC and joint.name == "Slider 29"
            )
            if is_revolute or is_master_slider:
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
        "Revolute 13": np.deg2rad(0.0),
        "Revolute 14": np.deg2rad(122.0),
        "Revolute 15": np.deg2rad(51.4),
        "Revolute 16": np.deg2rad(90.0),
        "Revolute 19": np.deg2rad(90.6),
        "Slider 29": -0.0,
    }
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

    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Joint Control")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    print_btn_rect = pygame.Rect(20, 60 + (len(controlled_dofs) + 2) * 60, 100, 30)
    print_requested = False

    sliders = [
        Slider(
            20,
            60 + i * 60,
            280,
            30,
            d["min"],
            d["max"],
            d["name"],
            d["is_rotational"],
            initial_value=default_vals.get(d["name"], (d["min"] + d["max"]) / 2),
        )
        for i, d in enumerate(controlled_dofs)
    ]
    sliders.append(box_pos_slider_x)
    sliders.append(box_pos_slider_y)

    target_positions = np.zeros(robot.n_dofs)
    box_pos = np.array([0.0, -0.1, 0.0])

    running = True
    while running:
        dialog_slider = None  # slider that wants to open a dialog this frame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if print_btn_rect.collidepoint(event.pos):
                    print_requested = True
            for slider in sliders:
                if slider.handle_event(event):
                    dialog_slider = slider  # only one can fire per frame

        # Open dialog outside the event loop, after all events are consumed.
        # scene.step() is NOT called while we are blocked in the dialog.
        if dialog_slider is not None:
            dialog_slider.open_dialog()

        for i, slider in enumerate(sliders[: len(controlled_dofs)]):
            target_positions[controlled_dofs[i]["idx"]] = slider.value

        if print_requested:
            for i, slider in enumerate(sliders[: len(controlled_dofs)]):
                print(
                    f"{controlled_dofs[i]['name']}: {slider.to_display_value(slider.value):.3f}{slider.unit}"
                )
            for s in sliders[len(controlled_dofs) :]:
                print(f"{s.name}: {s.to_display_value(s.value):.3f}{s.unit}")
            print_requested = False

        box_pos[0] = box_pos_slider_x.value
        box_pos[1] = box_pos_slider_y.value
        box.set_pos(box_pos)
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

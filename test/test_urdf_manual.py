import numpy as np
import genesis as gs
import pygame

WINDOW_SIZE = (320, 600)
FPS = 60


class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, name, is_rotational=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.name = name
        self.is_rotational = is_rotational
        self.value = 0.0
        self.dragging = False
        self.last_mouse_x = 0
        self.button_w = 10

    def to_display_value(self, val):
        if self.is_rotational:
            return np.rad2deg(val)
        return val

    def from_display_value(self, val):
        if self.is_rotational:
            return np.deg2rad(val)
        return val

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self.last_mouse_x = event.pos[0]
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                delta_x = event.pos[0] - self.last_mouse_x
                self.last_mouse_x = event.pos[0]
                range_val = self.max_val - self.min_val
                pixels_per_unit = self.rect.w / range_val if range_val != 0 else 1
                delta_val = delta_x / pixels_per_unit
                new_value = self.value + delta_val
                self.value = max(self.min_val, min(self.max_val, new_value))

    def draw(self, surface, font):
        pygame.draw.rect(surface, (128, 128, 128), self.rect)
        t = (
            (self.value - self.min_val) / (self.max_val - self.min_val)
            if self.max_val != self.min_val
            else 0
        )
        thumb_x = self.rect.x + int(t * (self.rect.w - self.button_w))
        thumb_rect = pygame.Rect(thumb_x, self.rect.y, self.button_w, self.rect.h)
        pygame.draw.rect(surface, (255, 255, 255), thumb_rect)
        dv = self.to_display_value(self.value)
        unit = "deg" if self.is_rotational else "m"
        value_str = f"{dv:.1f}{unit}"
        label = font.render(f"{self.name}: {value_str}", True, (0, 0, 0))
        surface.blit(label, (self.rect.x, self.rect.y - 20))


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

    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Joint Control")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    sliders = [
        Slider(
            20, 60 + i * 60, 280, 30, d["min"], d["max"], d["name"], d["is_rotational"]
        )
        for i, d in enumerate(controlled_dofs)
    ]

    target_positions = np.zeros(robot.n_dofs)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            for slider in sliders:
                slider.handle_event(event)

        for slider in sliders:
            target_positions[controlled_dofs[sliders.index(slider)]["idx"]] = (
                slider.value
            )

        robot.control_dofs_position(target_positions)
        scene.step()

        screen.fill((255, 255, 255))
        title = font.render("Joint Position Controls", True, (0, 0, 0))
        screen.blit(title, (20, 20))
        for slider in sliders:
            slider.draw(screen, font)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()

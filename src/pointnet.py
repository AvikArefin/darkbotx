import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from dataclasses import dataclass
import math
from stl import mesh
import mapbox_earcut as earcut
import trimesh
from urdf_parser_py.urdf import (
    Robot,
    Link,
    Inertial,
    Visual,
    Collision,
    Mesh,
    Inertia,
    Pose,
)


@dataclass
class UnitData:
    angle: np.float16
    width: np.float16
    side: str  # "both", "left", or "right"


class DarkBot:
    def __init__(
        self, measurements: list[tuple[float, float, str]], height: float
    ) -> None:
        """
        :param measurements: List of (angle, width, side)
                             side options: "left", "right", "both"
        """
        self.height = height
        self.data = [
            UnitData(np.float16(a), np.float16(w), s.lower())
            for a, w, s in measurements
        ]

    def get_all_endpoints(self) -> np.ndarray:
        pts = []
        for unit in self.data:
            angle_rad = np.deg2rad(float(unit.angle))
            # 'Width' is treated as the full span; hw is the radial distance from center
            hw = float(unit.width) / 2

            # "Left" point (Positive Vector)
            if unit.side in ["left", "both"]:
                pts.append([hw * np.cos(angle_rad), hw * np.sin(angle_rad)])

            # "Right" point (Negative/Mirrored Vector)
            if unit.side in ["right", "both"]:
                pts.append([-hw * np.cos(angle_rad), -hw * np.sin(angle_rad)])

        return np.array(pts)

    def getDentedBoundary(self) -> np.ndarray:
        pts = self.get_all_endpoints()
        if len(pts) < 3:
            return pts
        # Sort by polar angle to ensure the perimeter connects correctly
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        return pts[np.argsort(angles)]

    def getConvexHull(self) -> np.ndarray:
        pts = self.get_all_endpoints()
        if len(pts) < 3:
            return pts
        hull = ConvexHull(pts)
        return pts[hull.vertices]

    def visualize(self) -> None:
        plt.figure(figsize=(10, 10))

        all_pts = self.get_all_endpoints()
        if len(all_pts) == 0:
            print("No points to visualize.")
            return

        # Dynamically calculate the limit to keep (0,0) at the center
        # We find the furthest point from origin and add 20% padding
        max_dist = np.max(np.linalg.norm(all_pts, axis=1))
        plot_limit = max_dist * 1.2
        x_vals = np.linspace(-plot_limit, plot_limit, 100)

        # 1. Draw Sensor Walls & Segments
        for unit in self.data:
            angle_rad = np.deg2rad(float(unit.angle))
            hw = float(unit.width) / 2

            active_pts = []
            if unit.side in ["left", "both"]:
                active_pts.append(
                    np.array([hw * np.cos(angle_rad), hw * np.sin(angle_rad)])
                )
            if unit.side in ["right", "both"]:
                # The 'right' sensor is always 180 degrees opposite the 'left'
                active_pts.append(
                    np.array([-hw * np.cos(angle_rad), -hw * np.sin(angle_rad)])
                )

            # Draw the Blue width segments
            if len(active_pts) == 2:
                plt.plot(
                    [active_pts[0][0], active_pts[1][0]],
                    [active_pts[0][1], active_pts[1][1]],
                    "b-",
                    alpha=0.4,
                    linewidth=2,
                )
            elif len(active_pts) == 1:
                plt.plot(
                    [0, active_pts[0][0]],
                    [0, active_pts[0][1]],
                    "b--",
                    alpha=0.2,
                    linewidth=1,
                )

            # Draw Red perpendicular walls
            for pt in active_pts:
                # The wall slope is perpendicular to the sensor vector
                perp_angle = float(unit.angle) + 90
                m = math.tan(np.deg2rad(perp_angle))

                if abs(m) > 1e5:  # Vertical line
                    plt.axvline(x=pt[0], color="red", linestyle="--", alpha=0.15)
                else:
                    y_perp = m * (x_vals - pt[0]) + pt[1]
                    plt.plot(x_vals, y_perp, "r--", alpha=0.15)

        # 2. Draw Convex Hull (Green)
        hull_pts = self.getConvexHull()
        if len(hull_pts) >= 3:
            hull_loop = np.vstack([hull_pts, hull_pts[0]])
            plt.fill(
                hull_loop[:, 0], hull_loop[:, 1], "g", alpha=0.05, label="Convex Hull"
            )
            plt.plot(hull_loop[:, 0], hull_loop[:, 1], "g--", alpha=0.3, linewidth=1)

        # 3. Draw Dented Boundary (Orange)
        dented_pts = self.getDentedBoundary()
        if len(dented_pts) >= 3:
            dented_loop = np.vstack([dented_pts, dented_pts[0]])
            plt.fill(
                dented_loop[:, 0],
                dented_loop[:, 1],
                "orange",
                alpha=0.3,
                label="Dented Shell",
            )
            plt.plot(dented_loop[:, 0], dented_loop[:, 1], "orange", linewidth=2.5)

        # 4. Markers and Origin
        plt.scatter(all_pts[:, 0], all_pts[:, 1], color="black", s=40, zorder=10)
        plt.scatter(
            [0],
            [0],
            color="red",
            marker="x",
            s=100,
            label="Origin (Sensor Center)",
            zorder=11,
        )

        # Ensure centering by setting symmetric limits
        plt.xlim(-plot_limit, plot_limit)
        plt.ylim(-plot_limit, plot_limit)

        plt.title(
            f"DarkBot Visualization\nCentered on Sensor Origin (Points: {len(all_pts)})"
        )
        plt.gca().set_aspect("equal", adjustable="box")
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.legend(loc="upper right")
        plt.show()


def export_dented_to_stl(bot: DarkBot, filename: str):
    points_2d = bot.getDentedBoundary()
    height = bot.height
    pts_64 = points_2d.astype(np.float64)
    n = len(pts_64)

    # Triangulate caps
    rings = np.array([n], dtype=np.uint32)
    indices = earcut.triangulate_float64(pts_64, rings)

    num_cap_triangles = len(indices) // 3
    num_side_triangles = n * 2
    total_triangles = (num_cap_triangles * 2) + num_side_triangles

    data = np.zeros(total_triangles, dtype=mesh.Mesh.dtype)
    shape_mesh = mesh.Mesh(data)

    bottom_pts = np.column_stack([pts_64, np.zeros(n)])
    top_pts = np.column_stack([pts_64, np.full(n, height)])

    idx = 0
    # Create Side Walls
    for i in range(n):
        next_i = (i + 1) % n
        shape_mesh.vectors[idx] = [bottom_pts[i], top_pts[next_i], top_pts[i]]
        shape_mesh.vectors[idx + 1] = [
            bottom_pts[i],
            bottom_pts[next_i],
            top_pts[next_i],
        ]
        idx += 2

    # Create Caps
    for i in range(0, len(indices), 3):
        i1, i2, i3 = indices[i], indices[i + 1], indices[i + 2]
        shape_mesh.vectors[idx] = [top_pts[i1], top_pts[i2], top_pts[i3]]
        shape_mesh.vectors[idx + 1] = [bottom_pts[i1], bottom_pts[i3], bottom_pts[i2]]
        idx += 2

    shape_mesh.save(filename)
    print(f"Exported to {filename}")


def generate_urdf(stl_filepath, name, scale, urdf_filepath):
    # calculate inertia
    mesh = trimesh.load(stl_filepath)

    assert isinstance(mesh, trimesh.Trimesh)
    print("watertight: ", mesh.is_watertight)

    mesh.apply_scale(scale)
    mesh.density = 1250
    I = mesh.moment_inertia

    print(I)
    print("Volume: ", mesh.volume)
    print("Mass: ", mesh.mass)

    # urdf
    robot = Robot(name=name)
    link = Link(name="base")

    link.inertial = Inertial(
        mass=float(mesh.mass),
        inertia=Inertia(
            ixx=I[0, 0],
            iyy=I[1, 1],
            izz=I[2, 2],
            ixy=I[0, 1],
            ixz=I[0, 2],
            iyz=I[1, 2],
        ),
        origin=Pose(xyz=[0, 0, 0], rpy=[0, 0, 0]),
    )

    link.visual = Visual(
        geometry=Mesh(filename=stl_filepath, scale=[scale, scale, scale]),
        origin=Pose(xyz=[0, 0, 0], rpy=[0, 0, 0]),
    )

    link.collision = Collision(
        geometry=Mesh(filename=stl_filepath, scale=[scale, scale, scale]),
        origin=Pose(xyz=[0, 0, 0], rpy=[0, 0, 0]),
    )

    robot.add_link(link)

    # saving urdf
    with open(urdf_filepath, "w") as f:
        f.write(robot.to_xml_string())

    print(f"[SUCCESS] {urdf_filepath} is generated")


# --- EXAMPLE USAGE: TRIANGLE ---
if __name__ == "__main__":
    # obj_name = "triangle_prism"

    # # # To get a 3-sided triangle, provide 3 angles and pick ONE side.
    # # # Angle 90 (Top), 210 (Bottom-Left), 330 (Bottom-Right)
    # measurements = [
    #     (90,  6.0, "left"),
    #     (210, 6.0, "left"),
    #     (330, 6.0, "left")
    # ]
    # height_val = 4.0

    # obj_name = "cylinder"
    # measurements = [
    #     (0,   5.0, "both"),
    #     (30,  5.0, "both"),
    #     (60,  5.0, "both"),
    #     (90,  5.0, "both"),
    #     (120, 5.0, "both"),
    #     (150, 5.0, "both"),
    # ]
    # height_val = 5.7

    # 5.5cm Cube at 15-degree resolution
    obj_name = "cube"
    measurements = [
        (0, 5.50, "both"),
        (15, 5.70, "both"),
        (30, 6.35, "both"),
        (45, 7.78, "both"),
        (60, 6.35, "both"),
        (75, 5.70, "both"),
        (90, 5.50, "both"),
        (105, 5.70, "both"),
        (120, 6.35, "both"),
        (135, 7.78, "both"),
        (150, 6.35, "both"),
        (165, 5.70, "both"),
    ]
    height_val = 5.5

    darkbot = DarkBot(measurements, height=height_val)

    folder_path = f"assets/{obj_name}"
    os.makedirs(os.path.join(folder_path, "meshes"), exist_ok=True)
    stl_path = os.path.join(folder_path, "meshes", f"{obj_name}.stl")
    urdf_path = os.path.join(folder_path, f"{obj_name}.urdf")

    export_dented_to_stl(darkbot, stl_path)
    generate_urdf(stl_path, obj_name, 0.1, urdf_path)
    darkbot.visualize()

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from dataclasses import dataclass
import math
from stl import mesh
import mapbox_earcut as earcut
import trimesh
from urdf_parser_py.urdf import Robot, Link, Inertial, Visual, Collision, Mesh, Inertia, Pose

@dataclass
class UnitData:
    angle: np.float16
    width: np.float16

class DarkBot:
    def __init__(self, measurements: list[tuple[float, float]], height: float) -> None:
        """
        Initializes DarkBot with raw measurement pairs and a fixed height.
        :param measurements: A list of (angle, width) tuples.
        :param height: The extrusion height for the 3D model.
        """
        self.height = height
        # Internal conversion to UnitData keeps the structure you like 
        # without the manual labor at the call site.
        self.data = [UnitData(np.float16(a), np.float16(w)) for a, w in measurements]

    def get_all_endpoints(self) -> np.ndarray:
        pts = []
        for unit in self.data:
            angle_rad = np.deg2rad(float(unit.angle))
            hw = float(unit.width) / 2
            pts.append([hw * np.cos(angle_rad), hw * np.sin(angle_rad)])
            pts.append([-hw * np.cos(angle_rad), -hw * np.sin(angle_rad)])
        return np.array(pts)

    def getDentedBoundary(self) -> np.ndarray:
        """Sorts all 2N endpoints by polar angle to form a concave perimeter."""
        pts = self.get_all_endpoints()
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        return pts[np.argsort(angles)]

    def getConvexHull(self) -> np.ndarray:
        """Calculates the outer 'rubber band' envelope."""
        pts = self.get_all_endpoints()
        if len(pts) < 3: return pts
        hull = ConvexHull(pts)
        return pts[hull.vertices]

    def visualize(self) -> None:
        plt.figure(figsize=(10, 10))
        x_limit = 2.5
        x_vals = np.linspace(-x_limit, x_limit, 100)
        
        # 1. Draw the "Sensor Walls" (Parallel Lines)
        for unit in self.data:
            angle_rad = np.deg2rad(float(unit.angle))
            hw = float(unit.width) / 2
            
            # The two endpoints
            p1 = np.array([hw * np.cos(angle_rad), hw * np.sin(angle_rad)])
            p2 = -p1
            
            # Plot the width segment (The measurement itself)
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', alpha=0.4, linewidth=2)
            
            # Perpendicular lines at the endpoints
            for pt in [p1, p2]:
                perp_angle = float(unit.angle) + 90
                m = math.tan(np.deg2rad(perp_angle))
                
                if abs(m) > 1e5: # Vertical line
                    plt.axvline(x=pt[0], color='red', linestyle='--', alpha=0.15)
                else:
                    y_perp = m * (x_vals - pt[0]) + pt[1]
                    plt.plot(x_vals, y_perp, 'r--', alpha=0.15)

        # 2. Draw Convex Hull (The 'Maximum' Shell)
        hull_pts = self.getConvexHull()
        hull_loop = np.vstack([hull_pts, hull_pts[0]])
        plt.fill(hull_loop[:, 0], hull_loop[:, 1], 'g', alpha=0.08, label="Convex Hull")
        plt.plot(hull_loop[:, 0], hull_loop[:, 1], 'g--', alpha=0.4, linewidth=1)

        # 3. Draw Dented Boundary (The 'Actual' Shape)
        dented_pts = self.getDentedBoundary()
        dented_loop = np.vstack([dented_pts, dented_pts[0]])
        plt.fill(dented_loop[:, 0], dented_loop[:, 1], 'orange', alpha=0.3, label="Dented Shell")
        plt.plot(dented_loop[:, 0], dented_loop[:, 1], 'orange', linewidth=2.5)
        
        # 4. Cleanup and Markers
        all_pts = self.get_all_endpoints()
        plt.scatter(all_pts[:, 0], all_pts[:, 1], color='black', s=30, zorder=5)
        
        plt.title("DarkBot: Full Visualization\n(Width Segments, Boundary Walls, Convex Hull, and Dented Shell)")
        plt.xlim(-x_limit, x_limit)
        plt.ylim(-x_limit, x_limit)
        plt.axis('equal')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend(loc='upper right')
        plt.show()

def export_dented_to_stl(bot: DarkBot, filename: str = "output.stl"):
    points_2d = bot.getDentedBoundary()
    height = bot.height
    
    pts_64 = points_2d.astype(np.float64)
    if pts_64.ndim == 1:
        pts_64 = pts_64.reshape(-1, 2)
    
    n = len(pts_64)
    
    # Triangulate
    rings = np.array([n], dtype=np.uint32) 
    
    # Pass the (n, 2) array directly
    indices = earcut.triangulate_float64(pts_64, rings)
    
    # Setup Mesh
    num_cap_triangles = len(indices) // 3
    num_side_triangles = n * 2
    total_triangles = (num_cap_triangles * 2) + num_side_triangles
    
    data = np.zeros(total_triangles, dtype=mesh.Mesh.dtype)
    shape_mesh = mesh.Mesh(data)
    
    bottom_pts = np.column_stack([pts_64, np.zeros(n)])
    top_pts = np.column_stack([pts_64, np.full(n, height)])
    
    idx = 0
    # 4. Create Side Walls
    for i in range(n):
        next_i = (i + 1) % n
        shape_mesh.vectors[idx] = [bottom_pts[i], top_pts[next_i], top_pts[i]]
        shape_mesh.vectors[idx + 1] = [bottom_pts[i], bottom_pts[next_i], top_pts[next_i]]
        idx += 2
        
    # 5. Create Caps
    for i in range(0, len(indices), 3):
        i1, i2, i3 = indices[i], indices[i+1], indices[i+2]
        # Top
        shape_mesh.vectors[idx] = [top_pts[i1], top_pts[i2], top_pts[i3]]
        # Bottom (Flipped)
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
        mass = float(mesh.mass),
        inertia=Inertia(
            ixx = I[0, 0],
            iyy = I[1, 1],
            izz = I[2, 2],
            ixy = I[0, 1],
            ixz = I[0, 2],
            iyz = I[1, 2],
        ),
        origin=Pose(xyz=[0, 0, 0], rpy=[0,0,0])
    )

    link.visual = Visual(
        geometry=Mesh(
            filename=stl_filepath, 
            scale=[scale, scale, scale]
        ),
        origin=Pose(xyz=[0, 0, 0], rpy=[0,0,0])
    )

    link.collision = Collision(
        geometry=Mesh(
            filename=stl_filepath, 
            scale=[scale, scale, scale]
        ),
        origin=Pose(xyz=[0, 0, 0], rpy=[0,0,0])
    )

    robot.add_link(link)

    # saving urdf
    with open(urdf_filepath, "w") as f:
        f.write(robot.to_xml_string())

    print(f"[SUCCESS] {urdf_filepath} is generated")

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    obj_name = "cylinder"
    # Unified configuration
    measurements = [
        (0,   5.0),
        (30,  5.0),
        (60,  5.0),
        (90,  5.0),
        (120, 5.0),
        (150, 5.0),
    ]
    height_val = 5.7

    # obj_name = "cube"
    # Unified configuration
    # 5.5cm Cube at 15-degree resolution
    # measurements = [
    #     (0, 5.50), (15, 5.70), (30, 6.35), (45, 7.78), 
    #     (60, 6.35), (75, 5.70), (90, 5.50), (105, 5.70), 
    #     (120, 6.35), (135, 7.78), (150, 6.35), (165, 5.70)
    # ]
    # height_val = 5.5
    
    # Initialize with clean data
    darkbot = DarkBot(measurements, height=height_val)

    # File setup
    folder_path = f"assets/{obj_name}"
    os.makedirs(os.path.join(folder_path, "meshes"), exist_ok=True)
    stl_filepath = os.path.join(folder_path, "meshes", f"{obj_name}.stl")
    urdf_filepath = os.path.join(folder_path, f"{obj_name}.urdf")

    # Export & URDF
    export_dented_to_stl(darkbot, filename=stl_filepath)
    generate_urdf(stl_filepath, obj_name, 0.1, urdf_filepath)

    # visualize
    darkbot.visualize()

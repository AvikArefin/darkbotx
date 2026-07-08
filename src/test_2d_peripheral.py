import genesis as gs
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
import trimesh
import numpy as np

def get_urdf_meshes(urdf_path: str):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    base_dir = os.path.dirname(urdf_path)
    mesh_paths = []
    
    # Find all collision meshes first (or visual if no collision)
    for mesh in root.findall(".//collision/geometry/mesh") + root.findall(".//visual/geometry/mesh"):
        filename = mesh.get('filename')
        if filename:
            # Handle package:// or relative paths
            if filename.startswith("package://"):
                # simplified handling, strip package://
                filename = filename.replace("package://", "")
            
            full_path = os.path.join(base_dir, filename)
            scale_str = mesh.get('scale', '1.0 1.0 1.0')
            scale = np.array([float(s) for s in scale_str.split()])
            
            mesh_paths.append((full_path, scale))
            
    # Return unique paths
    return list({(p, tuple(s)) for p, s in mesh_paths})

def main():
    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False)
    
    urdf_path = os.path.join(os.path.dirname(__file__), "..", "assets", "star", "star.urdf")
    
    # Add a rotation to test the orientation projection
    morph = gs.morphs.URDF(file=urdf_path, euler=(0, 0, 30))  # 45 deg yaw (Genesis expects degrees)
    obj = scene.add_entity(morph)
    
    scene.build()
    
    # Get the actual pose of the object from genesis
    pos = obj.get_pos().cpu().numpy()
    quat = obj.get_quat().cpu().numpy()
    
    print(f"Parsing URDF to find mesh: {urdf_path}")
    meshes_info = get_urdf_meshes(urdf_path)
    
    if not meshes_info:
        print("No meshes found in URDF!")
        return
        
    mesh_file, scale = meshes_info[0]
    print(f"Found mesh: {mesh_file} with scale {scale}")
    
    # Load with trimesh
    t_mesh = trimesh.load(mesh_file)
    t_mesh.apply_scale(scale)
    
    # Construct transformation matrix from genesis pose
    # Genesis quat is (w, x, y, z), trimesh quaternion_matrix also expects (w, x, y, z)
    transform = trimesh.transformations.quaternion_matrix(quat)
    transform[0:3, 3] = pos
    t_mesh.apply_transform(transform)
    
    # Project to 2D
    print("Projecting mesh to 2D plane (XY)...")
    path2d = t_mesh.projected(normal=[0, 0, 1])
    
    # The projected path may consist of multiple discrete paths. We take the largest one (the outer boundary)
    discrete_paths = path2d.discrete
    if not discrete_paths:
        print("Could not extract a 2D path!")
        return
        
    # Usually the first one is the outer boundary, but let's take the one with the most points or largest area
    # In simple cases, there's just one main boundary.
    boundary_points = discrete_paths[0]
    
    # Number of distinct vertices is len(boundary_points) - 1 because the path is closed (first point == last point)
    num_vertices = len(boundary_points) - 1
    print(f"Extracted a true 2D periphery with {num_vertices} points!")
    print(boundary_points)
    
    # Plot the exact points
    x = boundary_points[:, 0]
    y = boundary_points[:, 1]
    
    # Plotting
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b', linewidth=2, label=f'True Periphery ({num_vertices} points)')
    plt.plot(0, 0, marker='x', color='r', markersize=10, label='Center')
    
    # Also plot the AABB just for comparison
    aabb = obj.get_AABB()
    if aabb.dim() == 3:
        size = aabb[0, 1] - aabb[0, 0]
    else:
        size = aabb[1] - aabb[0]
    lx, ly = size[0].item(), size[1].item()
    
    aabb_x = [lx/2, -lx/2, -lx/2, lx/2, lx/2]
    aabb_y = [ly/2, ly/2, -ly/2, -ly/2, ly/2]
    plt.plot(aabb_x, aabb_y, linestyle='--', color='gray', label='AABB (Bounding Box)')
    
    plt.title(f'Exact 2D Footprint Extracted from Mesh')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    print("\nDisplaying plot. Close the window to exit.")
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()

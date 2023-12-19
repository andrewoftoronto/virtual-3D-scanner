import open3d as o3d
import numpy as np

def run(clouds, points, colours, normals):

    # Stack points, normals, and colors into a single array

    #keeps = np.random.uniform(size=points.shape) < 0.5
    #points = points[keeps]
    #colours = colours[keeps]
    #normals = normals[keeps]

    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(points)
    #pcd.normals = o3d.utility.Vector3dVector(normals)
    #pcd.colors = o3d.utility.Vector3dVector(colours)

    #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1e-1, max_nn=30))

    # Create arrow geometries for normals
    '''arrow_scale = 0.02
    arrow_list = []
    for t in range(0, np.asarray(pcd.normals).shape[0], 4096):
        i = np.random.randint(np.asarray(pcd.normals).shape[0])
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=arrow_scale * 2.5,
            cone_radius=arrow_scale * 5,
            cone_height=arrow_scale * 2
        )
        arrow.paint_uniform_color([0, 0, 0])  # Set arrow color (e.g., red)
        arrow.translate(np.asarray(pcd.points)[i])
        arrow.rotate(
            R=look_at(pcd.normals[i], np.array([0, 1.0, 0])).T,
            center=np.asarray(pcd.points)[i]
        )
        arrow.scale(0.1, center=np.asarray(pcd.points)[i])
        arrow_list.append(arrow)'''

    o3d.visualization.draw_geometries(clouds)
    exit(0)
    print("Generating mesh")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh])

    print("Done")

def look_at(forward, world_up):
    forward = forward / np.linalg.norm(forward)
    right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)

    # Create a 3x3 view matrix
    forward = np.reshape(forward, [3, 1])
    right = np.reshape(right, [3, 1])
    up = np.reshape(up, [3, 1])
    view_matrix = np.concatenate((-right, -up, forward), axis=1)
    return view_matrix
import open3d as o3d
import numpy as np

def run(points, colours, normals):

    # Stack points, normals, and colors into a single array

    #keeps = np.random.uniform(size=points.shape) < 0.5
    #points = points[keeps]
    #colours = colours[keeps]
    #normals = normals[keeps]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors = o3d.utility.Vector3dVector(colours)
    o3d.visualization.draw_geometries([pcd])

    # Create a Trimesh object using the poisson method
    radii = [0.01, 0.05, 0.1, 0.2, 0.3]
    print("Generating mesh")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh])

    print("Done")
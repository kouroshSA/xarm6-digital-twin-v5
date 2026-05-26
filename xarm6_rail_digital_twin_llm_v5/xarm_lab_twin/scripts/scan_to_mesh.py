# scripts/scan_to_mesh.py
"""
Convert a point cloud scan to a watertight OBJ mesh suitable for MuJoCo.
Requires open3d: pip install open3d
"""
import open3d as o3d
import numpy as np
import sys


def process_scan(input_path: str, output_obj: str, voxel_size: float = 0.002):
    print(f"Loading scan: {input_path}")
    pcd = o3d.io.read_point_cloud(input_path)
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(100)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9
    )
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.02))
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=50000)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(output_obj, mesh)
    print(f"Saved: {output_obj}  ({len(mesh.triangles)} triangles)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/scan_to_mesh.py <input.ply> <output.obj>")
        sys.exit(1)
    process_scan(sys.argv[1], sys.argv[2])

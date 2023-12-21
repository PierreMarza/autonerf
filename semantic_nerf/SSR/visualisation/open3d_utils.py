################################################################
# Code adapted from https://github.com/Harry-Zhi/semantic_nerf #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)          #
################################################################

import numpy as np
import open3d as o3d


def trimesh_to_open3d(src):
    dst = o3d.geometry.TriangleMesh()
    dst.vertices = o3d.utility.Vector3dVector(src.vertices)
    dst.triangles = o3d.utility.Vector3iVector(src.faces)
    vertex_colors = src.visual.vertex_colors[:, :3].astype(np.float) / 255.0
    dst.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    dst.compute_vertex_normals()

    return dst


def clean_mesh(o3d_mesh, keep_single_cluster=False, min_num_cluster=200):
    import copy

    o3d_mesh_clean = copy.deepcopy(o3d_mesh)
    # http://www.open3d.org/docs/release/tutorial/geometry/mesh.html?highlight=cluster_connected_triangles
    (
        triangle_clusters,
        cluster_n_triangles,
        cluster_area,
    ) = o3d_mesh_clean.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    if keep_single_cluster:
        # keep the largest cluster.!
        largest_cluster_idx = np.argmax(cluster_n_triangles)
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        o3d_mesh_clean.remove_triangles_by_mask(triangles_to_remove)
        o3d_mesh_clean.remove_unreferenced_vertices()
        print("Show mesh with largest cluster kept")
    else:
        # remove small clusters
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_num_cluster
        o3d_mesh_clean.remove_triangles_by_mask(triangles_to_remove)
        o3d_mesh_clean.remove_unreferenced_vertices()
        print("Show mesh with small clusters removed")

    return o3d_mesh_clean

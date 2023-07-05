import torch
import torch.nn as nn
import pytorch3d.ops as ops

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	"""
	Compute the binary cross entropy loss for binary voxel grids.

	Args:
		voxel_src (torch.Tensor): Source voxel grid of shape (batch_size, height, width, depth).
		voxel_tgt (torch.Tensor): Target voxel grid of shape (batch_size, height, width, depth).

	Returns:
		torch.Tensor: Binary cross entropy loss.
	"""
	# Reshape the voxel tensors to 2D shape (batch_size, num_voxels)
	voxel_src_flat = voxel_src.view(voxel_src.size(0), -1)
	voxel_tgt_flat = voxel_tgt.view(voxel_tgt.size(0), -1)

	# Create BCE Loss instance
	criterion = nn.BCELoss()

	# Compute the loss
	loss = criterion(voxel_src_flat, voxel_tgt_flat)

	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	"""
	Compute the chamfer loss for point clouds.

	Args:
		point_cloud_src (torch.Tensor) : Source point cloud of shape (batch_size, n_points, xyz)
		point_cloud_tgt (torch.Tensor) : Target point cloud of shape (batch_size, n_points, xyz)

	Returns:
		torch.Tensor : Chamfer loss.

	"""

	# Compute pairwise distances
	dist_src_tgt, src_tgt_idx, _ = ops.knn.knn_points(point_cloud_src, point_cloud_tgt, K=1) # (batch_size, n_points, 1)
	dist_tgt_src, tgt_src_idx, _ = ops.knn.knn_points(point_cloud_tgt, point_cloud_src, K=1) # (batch_size, n_points, 1)

	# Compute chamfer loss
	loss_chamfer = torch.sum(torch.stack([dist_src_tgt, dist_tgt_src]))

	return loss_chamfer

def smoothness_loss(mesh_src):
    """
    Computes the Laplacian smoothing loss for a mesh.
    

    Args:
        mesh_src: A PyTorch3D Meshes object representing the source mesh.

    Returns:
        loss_laplacian: The computed Laplacian smoothing loss.

    Notes:
    - Involves computing the Laplacian operator on the vertex positions of the mesh 
	  and penalizing the difference between the Laplacian of the original mesh and 
	  the Laplacian of the smoothed mesh.
	- Laplacian operator is used to quantify the smoothness or regularity of a function or surface defined on a mesh.
      The Laplacian operator computes the difference between the vertex position and the average of its neighboring vertex positions, 
	  weighted by the adjacency relationships between vertices.
	- The Laplacian smoothing loss encourages smoother surfaces by penalizing
      abrupt changes or irregularities in the vertex positions.
    - The loss is calculated based on the squared Laplacian operator, which amplifies
      the impact of larger deviations from smoothness.
    - The Laplacian smoothing loss helps to promote visually appealing and coherent surfaces.
    """
    
	# mesh_src: Meshes object representing the source mesh

    # Get the vertex positions of the source mesh
    V = mesh_src.verts_packed()

    # Compute the Laplacian operator on the vertex positions
    L = mesh_src.laplacian_packed()

    # Compute the smoothness loss as the mean of the squared Laplacian operator
    loss_laplacian = torch.square(torch.linalg.norm(torch.matmul(L, V)))

    return loss_laplacian
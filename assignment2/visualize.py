import numpy as np
import matplotlib.pyplot as plt
import pytorch3d
from utils import get_device, get_mesh_renderer, get_points_renderer

"""
Functions to visualize result mesh, voxel, point clouds
"""

def visualize_mesh(mesh, output_path):
    '''
    Visualize a 3D mesh using PyTorch3D library 
    and save the rendered image to a specified output path
    Args:
        mesh_src (pytorch3d.structures.Meshes) : source mesh
        output_path (String) : path to output png file

    '''

    image_size = 512
    device = get_device()
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, 3.0]], device=device)

    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=1.5, elev=0.5, azim=90, at=((0, 0, 0), ))
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=T,  R=R, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    image = (rend*255).detach().cpu().numpy().astype(np.uint8).squeeze(0)
    plt.figure(figsize=(10, 10))
    plt.imshow(image[0, ..., :3])
    plt.imsave(output_path,image)
    plt.axis("off")

def visualize_voxel(voxels, output_path):
    '''
    Visualize a grid of binary voxels by turning into mesh 
    ,rendering mesh using PyTorch3D library 
    and saving the rendered image to a specified output path
    Args:
        voxels (torch.Tensor) : source voxel
        output_path (String) : path to output png file
    '''
    device = get_device()
    mesh = pytorch3d.ops.cubify(voxels, thresh=0.8).to(device)
    visualize_mesh(mesh, output_path)

def visualize_pcd(point_cloud_src, output_path):
    '''
    Visualize pointclouds using PyTorch3D library 
    and saving the rendered image to a specified output path
    Args:
        voxels (torch.Tensor) : source voxel
        output_path (String) : path to output png file
    '''
    device = get_device()
    points = point_cloud_src[0]
    color = (points - points.min()) / (points.max() - points.min())

    render_point_cloud = pytorch3d.structures.Pointclouds(
      points=[points], features=[color],
    ).to(device)

    renderer = get_points_renderer(image_size=256, device=device)
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=1.5, elev=0.5, azim=90, at=((0, 0, 0), ))
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=T,  R=R, device=device)
    rend = renderer(render_point_cloud, cameras=cameras)

    image = (rend*255).detach().cpu().numpy().astype(np.uint8).reshape((256,256,3))
    plt.figure(figsize=(10, 10))
    plt.imshow(image[0, ..., :3])
    plt.imsave(output_path,image)
    plt.axis("off")
    return
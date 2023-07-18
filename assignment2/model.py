from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

# reshape as layer
class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat: # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 1 x 32 x 32 x 32
            # pass
            # TODO:

            # g - did not work
            # self.decoder = nn.Sequential(
            #     nn.Linear(512, 32*32*32), # Fully connected layer to increase dimensionality
            #     nn.ReLU(),
            #     nn.Unflatten(1, (32,32,32)), # Reshape to 3D tensor (b x 32 x 32 x 32)
            #     nn.Sigmoid()  # Apply sigmoid activation to ensure voxel values between 0 and 1
            # )

            # self.decoder = nn.Sequential(
            #     nn.Linear(512, 2048),
            #     ReshapeLayer((-1, 256, 2, 2, 2)),
            #     nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=False, padding=1),
            #     nn.BatchNorm3d(128),
            #     nn.ReLU(),
            #     nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=False, padding=1),
            #     nn.BatchNorm3d(64),
            #     nn.ReLU(),
            #     nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1),
            #     nn.BatchNorm3d(32),
            #     nn.ReLU(),
            #     nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=False, padding=1),
            #     nn.BatchNorm3d(8),
            #     nn.ReLU(),
            #     nn.ConvTranspose3d(8, 1, kernel_size=1, bias=False),
            #     nn.Sigmoid()
            # )   

            # a -
            in_channels = 64
            channels = [32, 16, 8]
            modules = []
            for mid_channels in channels:
                modules += [
                    nn.ConvTranspose3d(in_channels=in_channels, out_channels=mid_channels, 
                                       kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm3d(mid_channels),
                    nn.ReLU()
                ]
                in_channels = mid_channels

            # output layer
            out_channels = 1
            modules += [
                nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(out_channels),
            ]
            self.decoder = torch.nn.Sequential(*modules)

        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            self.decoder = nn.Sequential(
                nn.Linear(512, self.n_point),
                nn.LeakyReLU(),
                nn.Linear(self.n_point, self.n_point*3),
                ReshapeLayer((-1, args.n_points, 3)),
                nn.Tanh()
            )
                     
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            self.decoder = nn.Sequential(
                nn.Linear(512, 3*mesh_pred.verts_packed().shape[0]),
                nn.Tanh()
            )              

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2)) # rearranges the dimensions to match the expected format (batch_size, channels, height, width)
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        ## check output size before moving forward!!
        if args.type == "vox":
            # TODO:
            # voxels_pred = self.decoder(encoded_feat)   
            # a - 
            voxels_pred = self.decoder(encoded_feat.reshape((encoded_feat.shape[0], 64, 2, 2, 2)))         
            return voxels_pred

        elif args.type == "point":
            # TODO:
            pointclouds_pred =  self.decoder(encoded_feat)   # 2, 5000, 3
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred = self.decoder(encoded_feat)             
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          


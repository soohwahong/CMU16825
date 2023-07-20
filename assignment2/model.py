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
    
# from https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/layers.py#L6
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx

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

            # g - did not work
            # self.decoder = nn.Sequential(
            #     nn.Linear(512, 32*32*32), # Fully connected layer to increase dimensionality
            #     nn.ReLU(),
            #     nn.Unflatten(1, (32,32,32)), # Reshape to 3D tensor (b x 32 x 32 x 32)
            #     nn.Sigmoid()  # Apply sigmoid activation to ensure voxel values between 0 and 1
            # )

            self.decoder = nn.Sequential(
                nn.Linear(512, 2048),
                ReshapeLayer((-1, 256, 2, 2, 2)),
                nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, bias=False, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(),
                nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, bias=False, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=False, padding=1),
                nn.BatchNorm3d(8),
                nn.ReLU(),
                nn.ConvTranspose3d(8, 1, kernel_size=1, bias=False),
                nn.Sigmoid()
            )   

            # # a -
            # in_channels = 64
            # channels = [32, 16, 8]
            # modules = []
            # for mid_channels in channels:
            #     modules += [
            #         nn.ConvTranspose3d(in_channels=in_channels, out_channels=mid_channels, 
            #                            kernel_size=4, stride=2, padding=1),
            #         nn.BatchNorm3d(mid_channels),
            #         nn.ReLU()
            #     ]
            #     in_channels = mid_channels

            # # output layer
            # out_channels = 1
            # modules += [
            #     nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels,
            #                        kernel_size=4, stride=2, padding=1),
            #     nn.BatchNorm3d(out_channels),
            # ]
            # self.decoder = torch.nn.Sequential(*modules)

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

        elif args.type == 'implicit':
            # Input : b x (512 + num_coords x 3)
            # Output : b x num_coords x 1
            
            # Start with a simple implementation of a network that predicts the occupancy given the image feature and a 3d coordinate as input. 
            # You will need to create a meshgrid of 32x32x32 in the normalized coordinate space of (-1,1)^3 to predict the full occupancy output.
            # It does not perform any form of normalization.
            

            self.decoder =  nn.Sequential(
                ResnetBlockFC(128, 128, size_h=None),
                ResnetBlockFC(128, 128, size_h=None),
                ResnetBlockFC(128, 128, size_h=None),
                ResnetBlockFC(128, 128, size_h=None),
                ResnetBlockFC(128, 128, size_h=None),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            # c_dim = 512, dim = 3
            # self.fc_p = nn.Linear(3, hidden_size)
            # self.fc_c = nn.Linear(512, hidden_size)
            # ResnetBlockFC(hidden_size)

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
            voxels_pred = self.decoder(encoded_feat)   
            # a - 
            # voxels_pred = self.decoder(encoded_feat.reshape((encoded_feat.shape[0], 64, 2, 2, 2)))         
            return voxels_pred

        elif args.type == "point":
            pointclouds_pred =  self.decoder(encoded_feat)   # 2, 5000, 3
            return pointclouds_pred

        elif args.type == "mesh":
            deform_vertices_pred = self.decoder(encoded_feat)             
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          
        
        elif args.type == 'implicit':
            # p : sampled points  (b x num_coords x 3)
            # c : conditioning input (2D image) (b x 512)
            # z : latent code? 

            self.n_coords = args.n_coords
            self.fc_p = nn.Linear(3*self.n_coords, 128)
            self.fc_c = nn.Linear(512, 128)

            net = self.fc_p(p)
            net_c = self.fc_c(encoded_feat).unsqueeze(1)

            net = net + net_c

            implicit_pred = self.decoder(net).squeeze(-1)

            return implicit_pred

# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
import os
import sys
import torch
# %matplotlib inline
# %matplotlib notebook
import os
import sys
import time
import json
import glob
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython import display
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import pytorch3d
sys.path.append('/home/neta-katz@staff.technion.ac.il/anaconda3/envs/pytorch3d')
# Data structures and functions for rendering
from pytorch3d.structures import Volumes
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)

# obtain the utilized device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    print(
        'Please note that NeRF is a resource-demanding method.'
        + ' Running this notebook on CPU will be extremely slow.'
        + ' We recommend running the example on a GPU'
        + ' with at least 10 GB of memory.'
    )
    device = torch.device("cpu")
from utils.generate_cow_renders import generate_cow_renders
from utils import image_grid
import torchvision
import torchvision.models as models
import pickle
from utils import plot_camera_scene
from pytorch3d.renderer.cameras import (
    SfMPerspectiveCameras,
)
from random import randrange

PATH = r'/home/neta-katz@staff.technion.ac.il/Downloads/'

# copying functions
def get_data(test_size=92):
    data_train_paths = [f for f in glob.glob(PATH +r'*.pkl')]
    # Do not put more than 1 pkl file!!!
    for path in data_train_paths:
        with open(path, 'rb') as outfile:
            x = pickle.load(outfile)
            # plot cameras
            R = np.array([r.T for r in x["cameras_R"]])
            R_test =R[:test_size]
            R_training = R[test_size:]
            T = x["cameras_pos"]
            T_test = T[:test_size]
            T_training = T[test_size:]
            cameras_absolute_gt = SfMPerspectiveCameras(
                R=R,
                T=x["cameras_pos"]/100,
                device=device,
            )
            plot_camera_scene(cameras_absolute_gt, cameras_absolute_gt, "HI")
            # plt.imshow(x["images"][0], interpolation='nearest', cmap='gray')
            cameras_training = FoVPerspectiveCameras(device=device, R=R_training, T=T_training)
            cameras_test = FoVPerspectiveCameras(device=device, R=R_test, T=T_test)
            # plt.imshow((np.stack((x["images"],)*3, axis=-1)*3)[0], interpolation='nearest', cmap='gray')
            temp_max = torch.from_numpy(np.stack((x["images"],)*3, axis=-1))[0].max()
            temp_min = torch.from_numpy(np.stack((x["images"],)*3, axis=-1))[0] .min()
            y = ((torch.from_numpy(np.stack((x["images"],)*3, axis=-1))[0] - temp_min)/(temp_max - temp_min))*temp_max
            # plt.imshow(y, interpolation='nearest', cmap='Dark2')
            # print(torch.from_numpy(np.stack((x["images"],)*3, axis=-1)).max())
            # print(torch.from_numpy(np.stack((x["images"],)*3, axis=-1)).min())
            # images_train, x_test = train_test_split(np.stack((x["images"],)*3, axis=-1), test_size=0.1)
            images = x["images"][0,:,0]
            images_test = images[:test_size]
            images_training = images[test_size:]

            images = torch.from_numpy(np.stack((images,)*3, axis=-1))
            # images = (images - images.min())/(images.max() - images.min())

            images_test = torch.from_numpy(np.stack((images_test,) * 3, axis=-1))
            images_test = images_test.float()
            images_test = (images_test - images.min()) / (images.max() - images.min())

            images_training = torch.from_numpy(np.stack((images_training,) * 3, axis=-1))
            images_training = images_training.float()
            images_training = (images_training - images.min()) / (images.max() - images.min())
            return cameras_training, images_training, cameras_test, images_test


cloud_training_cameras, cloud_training_images, cloud_test_cameras, cloud_test_images = get_data()

cloud_training_silhouettes = torch.zeros([1, 154, 154], dtype=torch.float32)
cloud_training_silhouettes[:] = cloud_training_images[:,:,:,0]

cloud_test_silhouettes = torch.zeros([92, 154, 154], dtype=torch.float32)
cloud_test_silhouettes[:] = cloud_test_images[:,:,:,0]

target_images = cloud_training_images
target_silhouettes = cloud_training_silhouettes
target_cameras = cloud_training_cameras

# get model
class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, omega0=0.1):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**(self.n_harmonic_functions-1) * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**(self.n_harmonic_functions-1) * x[..., i])
            ]

        Note that `x` is also premultiplied by `omega0` before
        evaluating the harmonic functions.
        """
        super().__init__()
        self.register_buffer(
            'frequencies',
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)

class NeuralRadianceField(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, n_hidden_neurons=256):
        super().__init__()
        """
        Args:
            n_harmonic_functions: The number of harmonic functions
                used to form the harmonic embedding of each point.
            n_hidden_neurons: The number of hidden units in the
                fully connected layers of the MLPs of the model.
        """

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)

        # The dimension of the harmonic embedding.
        embedding_dim = n_harmonic_functions * 2 * 3

        # self.mlp is a simple 2-layer multi-layer perceptron
        # which converts the input per-point harmonic embeddings
        # to a latent representation.
        # Not that we use Softplus activations instead of ReLU.
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
        )

        # Given features predicted by self.mlp, self.color_layer
        # is responsible for predicting a 3-D per-point vector
        # that represents the RGB color of the point.
        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons + embedding_dim, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, 3),
            torch.nn.Sigmoid(),
            # To ensure that the colors correctly range between [0-1],
            # the layer is terminated with a sigmoid layer.
        )

        # The density layer converts the features of self.mlp
        # to a 1D density value representing the raw opacity
        # of each point.
        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons, 1),
            torch.nn.Softplus(beta=10.0),
            # Sofplus activation ensures that the raw opacity
            # is a non-negative number.
        )

        # We set the bias of the density layer to -1.5
        # in order to initialize the opacities of the
        # ray points to values close to 0.
        # This is a crucial detail for ensuring convergence
        # of the model.
        self.density_layer[0].bias.data[0] = -1.5

    def _get_densities(self, features):
        """
        This function takes `features` predicted by `self.mlp`
        and converts them to `raw_densities` with `self.density_layer`.
        `raw_densities` are later mapped to [0-1] range with
        1 - inverse exponential of `raw_densities`.
        """
        raw_densities = self.density_layer(features)
        return 1 - (-raw_densities).exp()

    def _get_colors(self, features, rays_directions):
        """
        This function takes per-point `features` predicted by `self.mlp`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.

        In order to represent viewpoint dependent effects,
        before evaluating `self.color_layer`, `NeuralRadianceField`
        concatenates to the `features` a harmonic embedding
        of `ray_directions`, which are per-point directions
        of point rays expressed as 3D l2-normalized vectors
        in world coordinates.
        """
        spatial_size = features.shape[:-1]

        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(
            rays_directions, dim=-1
        )

        # Obtain the harmonic embedding of the normalized ray directions.
        rays_embedding = self.harmonic_embedding(
            rays_directions_normed
        )

        # Expand the ray directions tensor so that its spatial size
        # is equal to the size of features.
        rays_embedding_expand = rays_embedding[..., None, :].expand(
            *spatial_size, rays_embedding.shape[-1]
        )

        # Concatenate ray direction embeddings with
        # features and evaluate the color model.
        color_layer_input = torch.cat(
            (features, rays_embedding_expand),
            dim=-1
        )
        return self.color_layer(color_layer_input)

    def forward(
            self,
            ray_bundle: RayBundle,
            **kwargs,
    ):
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's
        RGB color and opacity respectively.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x 3]

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds = self.harmonic_embedding(
            rays_points_world
        )
        # embeds.shape = [minibatch x ... x self.n_harmonic_functions*6]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp(embeds)
        # features.shape = [minibatch x ... x n_hidden_neurons]

        # Finally, given the per-point features,
        # execute the density and color branches.

        rays_densities = self._get_densities(features)
        # rays_densities.shape = [minibatch x ... x 1]

        rays_colors = self._get_colors(features, ray_bundle.directions)
        # rays_colors.shape = [minibatch x ... x 3]

        return rays_densities, rays_colors

    def batched_forward(
            self,
            ray_bundle: RayBundle,
            n_batches: int = 16,
            **kwargs,
    ):
        """
        This function is used to allow for memory efficient processing
        of input rays. The input rays are first split to `n_batches`
        chunks and passed through the `self.forward` function one at a time
        in a for loop. Combined with disabling PyTorch gradient caching
        (`torch.no_grad()`), this allows for rendering large batches
        of rays that do not all fit into GPU memory in a single forward pass.
        In our case, batched_forward is used to export a fully-sized render
        of the radiance field for visualization purposes.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
            n_batches: Specifies the number of batches the input rays are split into.
                The larger the number of batches, the smaller the memory footprint
                and the lower the processing speed.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.

        """

        # Parse out shapes needed for tensor reshaping in this function.
        n_pts_per_ray = ray_bundle.lengths.shape[-1]
        spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]

        # Split the rays to `n_batches` batches.
        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), n_batches)

        # For each batch, execute the standard forward pass.
        batch_outputs = [
            self.forward(
                RayBundle(
                    origins=ray_bundle.origins.view(-1, 3)[batch_idx],
                    directions=ray_bundle.directions.view(-1, 3)[batch_idx],
                    lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx],
                    xys=None,
                )
            ) for batch_idx in batches
        ]

        # Concatenate the per-batch rays_densities and rays_colors
        # and reshape according to the sizes of the inputs.
        rays_densities, rays_colors = [
            torch.cat(
                [batch_output[output_i] for batch_output in batch_outputs], dim=0
            ).view(*spatial_size, -1) for output_i in (0, 1)
        ]
        return rays_densities, rays_colors


model = NeuralRadianceField().to(device)
model.load_state_dict(torch.load(r'/home/neta-katz@staff.technion.ac.il/Downloads/01_18_2023_13-11-37'))
model.eval()
#model = torch.load(r'/home/neta-katz@staff.technion.ac.il/Downloads/0'1_18_2023_11-29-22)

neural_radiance_field = model

# show test results
render_size = target_images.shape[1] * 2

# Our rendered scene is centered around (0,0,0)
# and is enclosed inside a bounding box
# whose side is roughly equal to 3.0 (world units).
volume_extent_world = 3.0
raysampler_grid = NDCMultinomialRaysampler(
    image_height=render_size,
    image_width=render_size,
    n_pts_per_ray=128,
    min_depth=0.1,
    max_depth=volume_extent_world,
)
raymarcher = EmissionAbsorptionRaymarcher()

renderer_grid = ImplicitRenderer(
    raysampler=raysampler_grid, raymarcher=raymarcher,
)
renderer_grid = renderer_grid.to(device)

def show_full_render_for_test(
        neural_radiance_field, camera,
        target_image, target_silhouette,
):
    """
    This is a helper function for visualizing the
    intermediate results of the learning.

    Since the `NeuralRadianceField` suffers from
    a large memory footprint, which does not let us
    render the full image grid in a single forward pass,
    we utilize the `NeuralRadianceField.batched_forward`
    function in combination with disabling the gradient caching.
    This chunks the set of emitted rays to batches and
    evaluates the implicit function on one batch at a time
    to prevent GPU memory overflow.
    """

    # Prevent gradient caching.
    with torch.no_grad():
        # Render using the grid renderer and the
        # batched_forward function of neural_radiance_field.
        rendered_image_silhouette, _ = renderer_grid(
            cameras=camera,
            volumetric_function=neural_radiance_field.batched_forward
        )
        # Split the rendering result to a silhouette render
        # and the image render.
        rendered_image, rendered_silhouette = (
            rendered_image_silhouette[0].split([3, 1], dim=-1)
        )

    # Generate plots.
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax = ax.ravel()
    clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
    ax[0].imshow(clamp_and_detach(rendered_image))
    ax[1].imshow(clamp_and_detach(rendered_silhouette[..., 0]))
    ax[2].imshow(clamp_and_detach(target_image))
    ax[3].imshow(clamp_and_detach(target_silhouette))
    for ax_, title_ in zip(
            ax,
            (
                    "rendered image", "rendered silhouette",
                    "target image", "target silhouette",
            )
    ):
        if not title_.startswith('loss'):
            ax_.grid("off")
            ax_.axis("off")
        ax_.set_title(title_)
    fig.canvas.draw();
    fig.show()
    display.clear_output(wait=True)
    display.display(fig)
    return fig

i = 0
for test_camera, test_img, test_sill in zip(cloud_test_cameras, cloud_test_images, cloud_test_silhouettes):
    print("image num", i)
    show_full_render_for_test(
        neural_radiance_field,
        FoVPerspectiveCameras(
            R=test_camera.R,
            T=test_camera.T,
            znear=test_camera.znear,
            zfar=test_camera.zfar,
            aspect_ratio=test_camera.aspect_ratio,
            fov=test_camera.fov,
            device=device,
        ),
        test_img,
        test_sill
    )
    i += 1

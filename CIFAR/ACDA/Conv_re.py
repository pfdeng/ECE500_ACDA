import torch
import torch.nn as nn
import torch.nn.functional as F


class ACDA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, spatial_atoms=4):
        super(ACDA, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_atoms = spatial_atoms

        # Filter-generating network
        self.filter_gen = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * spatial_atoms * kernel_size* kernel_size, kernel_size=1),
            nn.ReLU(inplace=True),
        )
         # Spatial position encoding
        self.position_encoding = torch.nn.Parameter(torch.empty(spatial_atoms, 2).uniform_(-1, 1), requires_grad=True)

        
    def forward(self, x):
        batch_size, _, height, width = x.size()

        # Generate dynamic filters for each pixel
        filters = self.filter_gen(x)
        filters = filters.view(batch_size, self.out_channels, self.spatial_atoms, self.kernel_size * self.kernel_size, height, width)
        
        # Create the position grid
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, height, device=x.device),
            torch.linspace(-1, 1, width, device=x.device),
            indexing='ij',
        )
        position_grid = torch.stack((x_grid, y_grid), dim=2).unsqueeze(0).to(x.device)

        # Calculate the spatial compositional coefficients
        spatial_att = torch.exp(-((position_grid.view(1, 1, height, width, 2) - self.position_encoding.view(1, self.spatial_atoms, 1, 1, 2)) ** 2).sum(dim=-1))

        # Combine filters and spatial compositional coefficients
        combined_filters = (filters * spatial_att.view(1, 1, self.spatial_atoms, 1, height, width)).sum(dim=2)
        combined_filters = combined_filters.view(batch_size, self.out_channels, self.kernel_size * self.kernel_size, height, width)



        # Perform adaptive convolution
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2)
        
        x_unfold = x_unfold.view(batch_size, self.out_channels, self.kernel_size * self.kernel_size, height, width)
        
        out = (combined_filters * x_unfold).sum(dim=2)
        out = out.view(batch_size, self.out_channels, height, width)

        return out



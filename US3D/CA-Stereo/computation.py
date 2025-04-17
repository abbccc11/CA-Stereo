import torch
import torch.nn as nn
import torch.nn.functional as F

class Estimation(nn.Module):
    def __init__(self, min_disp=-112.0, max_disp=16.0):
        super(Estimation, self).__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.conv = nn.Conv3d(
            in_channels=16,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,  # 'same' padding
            bias=True
        )

    def forward(self, inputs):
        """
        inputs: [N, 1, D, H, W]
        """
        # Apply 3D convolution
        x1 = self.conv(inputs)  # [N, 1, D, H, W]
        x = x1.squeeze(1)  # Remove channel dimension: [N, D, H, W]
        assert x.shape[1] == self.max_disp - self.min_disp, "Disparity range mismatch!"
        candidates = torch.linspace(
            self.min_disp,
            self.max_disp - 1.0,
            steps=self.max_disp - self.min_disp,
            device=x.device
        ).view(1, -1, 1, 1)

        # Compute probabilities using softmax
        probabilities = F.softmax(-1.0 * x, dim=1)  # Apply softmax along the disparity dimension

        # Compute the disparity map
        disparities = torch.sum(candidates * probabilities, dim=1, keepdim=True)  # [N, 1, H, W ]

        return x1, disparities
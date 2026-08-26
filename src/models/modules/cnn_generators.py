import torch
import torch.nn as nn


class CNNGenerator(nn.Module):
    """DCGAN-style conditional generator for 32×32 RGB images.

    Architecture: noise + label embedding → Linear → reshape (256, 4, 4)
    → three ConvTranspose2d blocks → (channels, 32, 32) with Tanh.
    """

    def __init__(
        self,
        n_classes: int,
        latent_dim: int,
        channels: int,
        img_size: int,
    ):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.init_size = img_size // 8  # 32 // 8 = 4

        self.fc = nn.Linear(latent_dim + n_classes, 256 * self.init_size ** 2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            # (256, 4, 4) → (128, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # (128, 8, 8) → (64, 16, 16)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # (64, 16, 16) → (channels, 32, 32)
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        gen_input = torch.cat((self.label_emb(labels), noise), dim=-1)
        out = self.fc(gen_input)
        out = out.view(out.size(0), 256, self.init_size, self.init_size)
        return self.conv_blocks(out)

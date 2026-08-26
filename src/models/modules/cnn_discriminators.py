import torch
import torch.nn as nn


class CNNDiscriminator(nn.Module):
    """DCGAN-style conditional discriminator for 32×32 RGB images.

    Label embedding is reshaped to (1, H, W) and concatenated as an extra
    channel so spatial conditioning information is preserved throughout
    the convolutional stack.

    Architecture: (channels+1, 32, 32) → three Conv2d blocks → Linear → 1.
    """

    def __init__(
        self,
        n_classes: int,
        channels: int,
        img_size: int,
    ):
        super().__init__()
        self.img_size = img_size
        self.label_emb = nn.Embedding(n_classes, img_size * img_size)

        self.model = nn.Sequential(
            # (channels+1, 32, 32) → (64, 16, 16)
            nn.Conv2d(channels + 1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 16, 16) → (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # (128, 8, 8) → (256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.adv_layer = nn.Linear(256 * (img_size // 8) ** 2, 1)

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Embed label → spatial map (batch, 1, H, W) then concat as extra channel
        label_map = self.label_emb(labels).view(img.size(0), 1, self.img_size, self.img_size)
        d_in = torch.cat((img, label_map), dim=1)
        out = self.model(d_in)
        return self.adv_layer(out.view(out.size(0), -1))

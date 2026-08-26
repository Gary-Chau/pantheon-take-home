from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torchmetrics import Accuracy
from torchvision.utils import make_grid

from src.models.mnist_gan_model import MNISTGANModel


class CIFAR10GANModel(MNISTGANModel):
    """Conditional GAN for CIFAR-10.

    Extends MNISTGANModel with:
    - torchmetrics discriminator accuracy tracked each validation epoch
    - RGB-aware image grid in on_epoch_end
    """

    def __init__(self, generator: nn.Module, discriminator: nn.Module, **kwargs):
        super().__init__(generator, discriminator, **kwargs)
        # Binary accuracy: real images labelled 1, fake images labelled 0
        self.val_disc_acc = Accuracy(task="binary")

    def validation_step(
        self, batch, batch_idx
    ) -> Union[Tensor, Dict[str, Any], None]:
        # Standard GAN loss logging (inherited step logic)
        log_dict, _ = self.step(batch, batch_idx)
        self.log_dict({"val/" + k: v for k, v in log_dict.items()})

        # Recompute discriminator outputs to update torchmetrics accuracy
        imgs, labels = batch
        batch_size = imgs.shape[0]
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        gen_labels = torch.randint(
            0, self.hparams.n_classes, (batch_size,), device=self.device
        )

        with torch.no_grad():
            gen_imgs = self.generator(z, gen_labels)
            validity_real = self.discriminator(imgs, labels)
            validity_fake = self.discriminator(gen_imgs, gen_labels)

        # Convert LSGAN outputs to [0,1] probabilities and build targets
        probs = torch.sigmoid(
            torch.cat([validity_real, validity_fake])
        ).squeeze(1)
        targets = torch.cat([
            torch.ones(batch_size, dtype=torch.long, device=self.device),
            torch.zeros(batch_size, dtype=torch.long, device=self.device),
        ])
        self.val_disc_acc(probs, targets)

        return None

    def on_validation_epoch_end(self) -> None:
        self.log("val/disc_acc", self.val_disc_acc.compute())
        self.val_disc_acc.reset()

    def on_epoch_end(self) -> None:
        # One sample per class in a 2-row × 5-col grid (same as parent but explicit RGB)
        z = torch.randn(self.hparams.n_classes, self.hparams.latent_dim, device=self.device)
        labels = torch.arange(self.hparams.n_classes, device=self.device)

        self.generator.eval()
        with torch.no_grad():
            gen_imgs = self.generator(z, labels)
        self.generator.train()

        # Rescale from Tanh output [-1, 1] to [0, 1]
        gen_imgs = (gen_imgs + 1) / 2
        grid = make_grid(gen_imgs, nrow=5, normalize=False)

        for logger in self.trainer.logger:
            if type(logger).__name__ == "WandbLogger":
                logger.experiment.log(
                    {
                        "gen_imgs": wandb.Image(
                            grid,
                            caption=f"Epoch {self.current_epoch} | CIFAR-10 classes 0-9",
                        )
                    },
                    step=self.current_epoch,
                )

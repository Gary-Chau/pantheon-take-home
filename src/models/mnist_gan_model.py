from typing import Union, Dict, Any, Tuple, Optional

import wandb
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.utils import make_grid
from pytorch_lightning import LightningModule


class MNISTGANModel(LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = generator
        self.discriminator = discriminator
        self.adversarial_loss = torch.nn.MSELoss()

    def forward(self, z, labels) -> Tensor:
        return self.generator(z, labels)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2)
        )
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx, optimizer_idx) -> Union[Tensor, Dict[str, Any]]:
        log_dict, loss = self.step(batch, batch_idx, optimizer_idx)
        self.log_dict({"/".join(("train", k)): v for k, v in log_dict.items()})
        return loss

    def validation_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        log_dict, loss = self.step(batch, batch_idx)
        self.log_dict({"/".join(("val", k)): v for k, v in log_dict.items()})
        return None

    def test_step(self, batch, batch_idx) -> Union[Tensor, Dict[str, Any], None]:
        log_dict, loss = self.step(batch, batch_idx)
        self.log_dict({"/".join(("test", k)): v for k, v in log_dict.items()})
        return None

    def step(self, batch, batch_idx, optimizer_idx=None) -> Tuple[Dict[str, Tensor], Optional[Tensor]]:
        imgs, labels = batch
        batch_size = imgs.shape[0]

        log_dict = {}
        loss = None

        # Adversarial ground truths
        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        # Noise and random labels used as generator input
        z = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)
        gen_labels = torch.randint(0, self.hparams.n_classes, (batch_size,), device=self.device)

        if optimizer_idx == 0 or not self.training:
            # Generate a batch of images
            gen_imgs = self.generator(z, gen_labels)

            # Calculate loss to measure generator's ability to fool the discriminator
            validity = self.discriminator(gen_imgs, gen_labels)
            g_loss = self.adversarial_loss(validity, valid)

            log_dict["g_loss"] = g_loss
            loss = g_loss

        if optimizer_idx == 1 or not self.training:
            # Generate a batch of images (detached so the generator is not updated here)
            gen_imgs = self.generator(z, gen_labels).detach()

            # Calculate loss for real images
            validity_real = self.discriminator(imgs, labels)
            d_real_loss = self.adversarial_loss(validity_real, valid)

            # Calculate loss for fake images
            validity_fake = self.discriminator(gen_imgs, gen_labels)
            d_fake_loss = self.adversarial_loss(validity_fake, fake)

            # Calculate total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            log_dict["d_loss"] = d_loss
            log_dict["d_real_loss"] = d_real_loss
            log_dict["d_fake_loss"] = d_fake_loss
            loss = d_loss

        return log_dict, loss

    def on_epoch_end(self):
        # One sample per class (0-9), fixed for consistent visual comparison across epochs
        z = torch.randn(self.hparams.n_classes, self.hparams.latent_dim, device=self.device)
        labels = torch.arange(self.hparams.n_classes, device=self.device)

        self.generator.eval()
        with torch.no_grad():
            gen_imgs = self.generator(z, labels)
        self.generator.train()

        # Rescale from Tanh output [-1, 1] to [0, 1] for visualisation
        gen_imgs = (gen_imgs + 1) / 2

        # Arrange into a 2-row × 5-col grid (one cell per digit class)
        grid = make_grid(gen_imgs, nrow=5, normalize=False)

        for logger in self.trainer.logger:
            if type(logger).__name__ == "WandbLogger":
                logger.experiment.log(
                    {
                        "gen_imgs": wandb.Image(
                            grid,
                            caption=f"Epoch {self.current_epoch} | rows: 0-4, 5-9",
                        )
                    },
                    step=self.current_epoch,
                )

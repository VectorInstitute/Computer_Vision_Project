"""
Contains pieces taken from https://github.com/pytorch/examples/issues/70
and https://github.com/pytorch/examples/tree/master/vae
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torchvision.utils import save_image
from vector_cv_tools import utils as vutils
from vector_cv_tools import transforms as VT
from vector_cv_tools import datasets as vdatasets

MVTEC_ROOT_DIR = "/scratch/ssd002/datasets/MVTec_AD"


def loss_fn(x, z, 
            lmdax, lmdaz, 
            x_recon_batch, z_recon_batch, 
            mu, logvar):

    B = x.size(0)

    MSEx = (x - x_recon_batch).pow(2).sum() / B
    MSEz = (z - z_recon_batch).pow(2).sum() / B
    Lx = lmdax * MSEx
    Lz = lmdaz * MSEz

    KLD = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()
    return Lx + Lz + KLD


class Decoder(nn.Module):

    def __init__(self):
        nc = 3
        ndf = 128
        ngf = 128
        nz = 100
        super().__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 128 x 128
        )

    def forward(self, x):
        return self.main(x)


class Encoder(nn.Module):

    def __init__(self):
        nc = 3
        ndf = 128
        ngf = 128
        nz = 100
        super().__init__()
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 2 * nz, 4, stride=1, padding=0, bias=False),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.main(x)


class ConvVAE(nn.Module):

    def __init__(self,lmda = 0.5):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
	self.set_lambda(lmda)

    def set_lambda(self,lmda):
        self.z_lambda = lmda
        self.x_lambda = 1 - lmda

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc_out = self.encoder(x)
        mu, logvar = enc_out[..., :100], enc_out[..., 100:]
        z = self.reparameterize(mu, logvar)
        x_recon_batch = self.decoder(z.unsqueeze(-1).unsqueeze(-1))
	z_recon_batch = self.encoder(x_recon_batch)
        return z, x_recon_batch, z_recon_batch, mu, logvar


def to_loader(dset, batch_size=128, num_workers=4):
    # note that this collate fn is needed for all our image datasets
    # as the PyTorch default WILL load the data in the wrong ordering
    return DataLoader(dset,
                      collate_fn=vutils.collate_dictionary_fn,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      pin_memory=True,
                      shuffle=True)


def train():

    lr = 3e-5 # parameter for Adam optimizer

    lmda = 0.5 # fishAE parameter

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    training_transforms = VT.ComposeMVTecTransform(
        [A.Resize(128, 128),
         A.ToFloat(max_value=255),
         ToTensorV2()])
    mvtec_train_dset = vdatasets.MVTec(MVTEC_ROOT_DIR,
                                       split="train",
                                       transforms=training_transforms)

    train_loader = to_loader(mvtec_train_dset)
    model = ConvVAE(lmda=lmda).to(device)
    model = torch.nn.DataParallel(model)


    optimizer = Adam(model.parameters(), lr=lr)
    num_epochs = 500

    for epoch in tqdm(range(num_epochs)):
        train_losses = []
        for i, (data, _) in enumerate(train_loader):
            x = data.to(device)
            optimizer.zero_grad()
            z, x_recon_batch, z_recon_batch, mu, logvar = model(x)
            loss = loss_fn(x, z,
                           self.x_lambda, self.z_lambda,
                           x_recon_batch, z_recon_batch,
                           mu, logvar)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if i % 100 == 0:
                save_image(recon.detach().cpu(), f"{epoch+1}_{i}.png")
                print(loss.item())
        mean_loss = sum(train_losses) / len(train_losses) if (
            len(train_losses) > 0) else 0
        print(f"train loss at epoch {epoch+1} is {mean_loss}")
        if (epoch % 20) == 0:
            torch.save(model.state_dict(), f"model_{epoch}.pt")


if __name__ == "__main__":
    train()

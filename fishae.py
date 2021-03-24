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
import sys
import os

MVTEC_ROOT_DIR = "/scratch/ssd002/datasets/MVTec_AD"


def loss_fn(x, z, 
            lmdax, lmdaz, 
            x_recon_batch, z_recon_batch, 
            ):

    B = x.size(0)

    MSEx = (x - x_recon_batch).pow(2).sum() / B
    MSEz = (z - z_recon_batch).pow(2).sum() / B
    Lx = lmdax * MSEx
    Lz = lmdaz * MSEz

    return Lx + Lz


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
            nn.Conv2d(ndf * 16, nz, 4, stride=1, padding=0, bias=False),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.main(x)


class FishAE(nn.Module):

    def __init__(self,lmda = 0.5):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.set_lambda(lmda)

    def set_lambda(self,lmda):
        self.z_lambda = lmda
        self.x_lambda = 1 - lmda

    def forward(self, x):
        z = self.encoder(x)
        x_recon_batch = self.decoder(z.unsqueeze(-1).unsqueeze(-1))
        z_recon_batch = self.encoder(x_recon_batch)
        return z, x_recon_batch, z_recon_batch


def to_loader(dset, batch_size=128, num_workers=4):
    # note that this collate fn is needed for all our image datasets
    # as the PyTorch default WILL load the data in the wrong ordering
    return DataLoader(dset,
                      collate_fn=vutils.collate_dictionary_fn,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      pin_memory=True,
                      shuffle=True)


def train(lmda = None):

    lr = 3e-5 # parameter for Adam optimizer

    if lmda is None: # wow, def remember to say 'is None' not just 'if lmda' if you want 0 to be valid...
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
    model = FishAE(lmda=lmda).to(device)
    model = torch.nn.DataParallel(model)

    # Checkpointage

    checkpointStr = f"fishae_v1_lmda{lmda}_"

    checkpointDir = "/checkpoint/ttrim/fae"
    print(f"Looking for files starting {checkpointStr} in {checkpointDir}")
    checkpointFilenames = os.listdir(checkpointDir)
    releventFilenames = tuple(filename for filename 
            in checkpointFilenames if filename.startswith(checkpointStr))

    if releventFilenames:

        print(f"Found the following:")
        print(releventFilenames)

        lastCheckpointFilename = max(releventFilenames)
        print(f"Restarting training from checkpoint {lastCheckpointFilename}")

        model.load_state_dict(torch.load(f"{checkpointDir}/{lastCheckpointFilename}"))
        # epoch is not 0!!!
        lastepo = int(lastCheckpointFilename.split("epo")[-1].strip(".pt"))
        print(f"last epoch: {lastepo}")
    else:
        print("... didn't find any")
        print(f"Beginning training for {checkpointStr}")
        lastepo = 0


    optimizer = Adam(model.parameters(), lr=lr)
    num_epochs = 501

    extra_checkpoints = (1,2,3,5,8,13)

    for epoch in tqdm(range(lastepo+1,num_epochs)):
        print(f"starting epoch {epoch}")
        train_losses = []
        for i, (data, _) in enumerate(train_loader):
            x = data.to(device)
            optimizer.zero_grad()
            z, x_recon_batch, z_recon_batch = model(x)
            loss = loss_fn(x, z,
                           model.module.x_lambda, model.module.z_lambda,
                           x_recon_batch, z_recon_batch,
                           )
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # these aren't even good.
            # maybe if they were more at start of training
            # and less later on. but as it stands, I'll just look at the checkpoints.
          #  if i % 100 == 0:
          #      save_image(recon.detach().cpu(), f"{epoch+1}_{i}.png")
          #      print(loss.item())

        mean_loss = sum(train_losses) / len(train_losses) if (
            len(train_losses) > 0) else 0
        print(f"train loss after epoch {epoch} is {mean_loss}")
        if ((epoch % 20) == 0
                or epoch in extra_checkpoints):
            torch.save(model.state_dict(), f"{checkpointDir}/{checkpointStr}epo{epoch}.pt")


if __name__ == "__main__":

    if len(sys.argv)==2:
        print(f"Recieved {sys.argv[1]} as argument. Setting lambda.")
        lmda = float(sys.argv[1])
    else:
        raise Exception("Missing required argument lambda.")
        lmda = 0.5

    train(lmda)

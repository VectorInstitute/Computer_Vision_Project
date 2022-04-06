import torch
import torch.nn as nn



class Decoder(nn.Module):
    """
    The model architecture is taken from https://github.com/pytorch/examples/issues/70
    """

    def __init__(self, in_channels, dec_channels, hidden_dim):
        self.in_channels = in_channels
        self.dec_channels = dec_channels
        self.hidden_dim = hidden_dim
        
        super().__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.hidden_dim, self.dec_channels * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.dec_channels * 16),
            nn.ReLU(True),
            # state size. (NGF*16) x 4 x 4
            nn.ConvTranspose2d(self.dec_channels * 16, self.dec_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dec_channels * 8),
            nn.ReLU(True),
            # state size. (NGF*8) x 8 x 8
            nn.ConvTranspose2d(self.dec_channels * 8, self.dec_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dec_channels * 4),
            nn.ReLU(True),
            # state size. (NGF*4) x 16 x 16
            nn.ConvTranspose2d(self.dec_channels * 4, self.dec_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dec_channels * 2),
            nn.ReLU(True),
            # state size. (NGF*2) x 32 x 32
            nn.ConvTranspose2d(self.dec_channels * 2, self.dec_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dec_channels),
            nn.ReLU(True),
            # state size. (NGF) x 64 x 64
            nn.ConvTranspose2d(self.dec_channels, self.in_channels, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (NC) x 128 x 128
        )

    def forward(self, x):
        return self.main(x)


class Encoder(nn.Module):
    """
    The model architecture is taken from https://github.com/pytorch/examples/issues/70
    """

    def __init__(self, in_channels, enc_channels, hidden_dim):
        self.in_channels = in_channels
        self.enc_channels = enc_channels
        self.hidden_dim = hidden_dim
        
        super().__init__()
        self.main = nn.Sequential(
            # input is (NC) x 128 x 128
            nn.Conv2d(self.in_channels, self.enc_channels, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF) x 64 x 64
            nn.Conv2d(self.enc_channels, self.enc_channels * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.enc_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*2) x 32 x 32
            nn.Conv2d(self.enc_channels * 2, self.enc_channels * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.enc_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*4) x 16 x 16
            nn.Conv2d(self.enc_channels * 4, self.enc_channels * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.enc_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*8) x 8 x 8
            nn.Conv2d(self.enc_channels * 8, self.enc_channels * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.enc_channels * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*16) x 4 x 4
            nn.Conv2d(self.enc_channels * 16, self.hidden_dim, 4, stride=1, padding=0, bias=False),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.main(x)


def vae_loss_fn(x, recon_batch, mu, logvar):
    """Function taken and modified from
        https://github.com/pytorch/examples/tree/master/vae
    """
    B = x.size(0)
    MSE = (x - recon_batch).pow(2).sum() / B
    KLD = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()
    return MSE + KLD

def ae_loss_fn(x, recon_batch):
    """Function taken and modified from
        https://github.com/pytorch/examples/tree/master/vae
    """
    B = x.size(0)
    MSE = (x - recon_batch).pow(2).sum() / B
    return MSE

class ConvVAE(nn.Module):

    def __init__(self, in_channels=3, enc_channels=128, dec_channels=128, hidden_dim=100):
        super().__init__()
        
        self.in_channels = in_channels
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        self.hidden_dim = hidden_dim
        
        self.encoder = Encoder(self.in_channels, self.enc_channels, self.hidden_dim*2)
        self.decoder = Decoder(self.in_channels, self.dec_channels, self.hidden_dim)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        
        return mu + eps * std

    def forward(self, x):
        enc_out = self.encoder(x)
        mu, logvar = enc_out[..., :self.hidden_dim], enc_out[..., self.hidden_dim:]
        z = self.reparameterize(mu, logvar)
        recon_batch = self.decoder(z.unsqueeze(-1).unsqueeze(-1))
        return recon_batch, mu, logvar

class AE(nn.Module):

    def __init__(self, in_channels=3, enc_channels=128, dec_channels=128, hidden_dim=100):
        super().__init__()
        self.in_channels = in_channels
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(self.in_channels, self.enc_channels, self.hidden_dim)
        self.decoder = Decoder(self.in_channels, self.dec_channels, self.hidden_dim)
                 
    def forward(self, x):
        enc_out = self.encoder(x)
        recon_batch = self.decoder(enc_out.unsqueeze(-1).unsqueeze(-1))
        return recon_batch

    
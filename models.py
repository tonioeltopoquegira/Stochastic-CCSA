
import torch
import torch.nn as nn
import torch.nn.functional as F

# MNIST small CNN 
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)   # 28->14
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 14 * 14, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# CIFAR ResNet
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32,  num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64,  num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
    def _make_layer(self, block, planes, nblocks, stride):
        strides = [stride] + [1]*(nblocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, stride=s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)

def resnet32(num_classes=10):
    n = 5   # (32-2)/6 = 5
    return ResNetCIFAR(BasicBlock, [n, n, n], num_classes=num_classes)

def resnet56(num_classes=100):
    n = 9   # (56-2)/6 = 9
    return ResNetCIFAR(BasicBlock, [n, n, n], num_classes=num_classes)


class ConvVAE(nn.Module):
    """
    Convolutional VAE suitable for 32x32 RGB (CIFAR) or 28x28 gray (MNIST).
    Reconstruction uses BCE on pixels in [0,1].
    """
    def __init__(self, input_channels=3, img_size=32, latent_dim=128, hidden_channels=128):
        super().__init__()
        self.input_channels = input_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels

        # Encoder: conv downsampling to feature map, then flatten -> mu/logvar
        # Simple 4-layer conv encoder for 32x32 -> small feature map
        self.enc = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels//2, 4, 2, 1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(hidden_channels//2, hidden_channels, 4, 2, 1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1),     # 8 -> 4
            nn.ReLU(),
        )
        # compute flattened size after convs for given img_size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, img_size, img_size)
            feat = self.enc(dummy)
            self._enc_flat_dim = int(feat.numel() // feat.size(0))

        self.fc_mu = nn.Linear(self._enc_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._enc_flat_dim, latent_dim)

        # Decoder: linear from z to feature map, then transposed convs
        self.fc_dec = nn.Linear(latent_dim, self._enc_flat_dim)
        # using mirrored convs
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1),  # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels//2, 4, 2, 1), # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels//2, input_channels, 4, 2, 1), # 16 -> 32
            nn.Sigmoid()  # output in [0,1]
        )

    def encode(self, x):
        h = self.enc(x)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(h.size(0), -1, int((self.img_size // 8)), int((self.img_size // 8)))
        # above assumes the enc output spatial size is img_size/8 (with our convs)
        out = self.dec(h)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss_components(recon_x, x, mu, logvar, reduction='mean'):
        """
        Returns (recon_loss, kl_loss)
        recon_loss: BCE between recon_x and x (images expected in [0,1])
        kl_loss: mean KL divergence per batch
        """
        recon_flat = recon_x.view(recon_x.size(0), -1)
        x_flat = x.view(x.size(0), -1)
        recon_loss = F.binary_cross_entropy(recon_flat, x_flat, reduction=reduction)
        kl_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_sample = kl_element.sum(dim=1)
        if reduction == "mean":
            kl_loss = kl_per_sample.mean()
        elif reduction == "sum":
            kl_loss = kl_per_sample.sum()
        else:
            kl_loss = kl_per_sample
        return recon_loss, kl_loss
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch.nn.functional import one_hot
from tqdm import tqdm
from models.Blocks import ConvBlock, TransposeConvBlock
import numpy as np

import matplotlib.pyplot as plt

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# channel of latent vector
z_dim = 100
n_classes = 10

def generate_example(G):
    noise = torch.randn(BATCH_SIZE, z_dim, device="cuda")
    noise = noise.view(BATCH_SIZE, z_dim, 1, 1)

    number = np.random.randint(0, 10)
    code1 = one_hot(torch.ones((BATCH_SIZE), dtype=int) * number, num_classes=n_classes).cuda()

    imgs = G(noise, code1.view(-1, n_classes, 1, 1))

    img = torchvision.utils.make_grid(imgs)

    return img


class Discriminator(nn.Module):
    def __init__(self, n_classes, hidden_c):
        super().__init__()
        # B x 1 x 28 x 28 -> B x 64 x 28 x 28
        self.input_layer = ConvBlock(in_channel=1, out_channel=hidden_c, ksize=5, stride=1)
        
        self.hidden_layers = nn.Sequential(
            # B x 64 x 28 x 28 -> B x 128 x 14 x 14
            ConvBlock(in_channel=1 * hidden_c, out_channel=2 * hidden_c, ksize=3, stride=1, downsample=True),
            # B x 128 x 14 x 14 -> B x 256 x 14 x 14 
            ConvBlock(in_channel=2 * hidden_c, out_channel=4 * hidden_c, ksize=3, stride=1),
            # B x 256 x 14 x 14 -> B x 512 x 7 x 7
            ConvBlock(in_channel=4 * hidden_c, out_channel=8 * hidden_c, ksize=3, stride=1, downsample=True),
        )
        self.output_layer = nn.Sequential(
            # B x 512 x 7 x 7 -> B x 100 x 7 x 7
            ConvBlock(in_channel=8 * hidden_c, out_channel=z_dim + n_classes, ksize=1, stride=1),
        )
        self.z_layer = nn.Sequential(
            # B x 100 x 7 x 7 -> B x 100 x 1 x 1
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # B x 100 x 1 x 1 -> B x 1
            nn.Linear(z_dim + n_classes, 1),
            nn.Sigmoid()
        )

        self.c_layer = nn.Sequential(
            # B x 100 -> B x 10 x 7 x 7
            nn.Conv2d(z_dim + n_classes, n_classes, 1, 1),
            nn.BatchNorm2d(10),
            # B x 10 x 7 x 7 -> B x 10 x 1 x 1
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_layers(out)
        out = self.output_layer(out)

        z = self.z_layer(out)
        c = self.c_layer(out)

        return z, c
        

class Generator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.input_layer = nn.Sequential(
            # B x 100 x 1 x 1 -> B x 100 x 7 x 7 
            TransposeConvBlock(z_dim + n_classes, z_dim + n_classes, ksize=7, stride=1),
            # B x 100 x 7 x 7 -> B x 512 x 7 x 7
            TransposeConvBlock(z_dim + n_classes, 8 * hidden_dim, upsample=False),
        )
        self.hidden_layers = nn.Sequential(
            # B x 512 x 7 x 7 -> B x 256 x 14 x 14
            TransposeConvBlock(8 * hidden_dim, 4 * hidden_dim, upsample=True),
            # B x 256 x 14 x 14 -> B x 128 x 14 x 14
            TransposeConvBlock(4 * hidden_dim, 2 * hidden_dim, upsample=False),
            # B x 128 x 14 x 14 -> B x 64 x 28 x 28
            TransposeConvBlock(2 * hidden_dim, 1 * hidden_dim, upsample=True),
        )
        self.output_layer = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x, c):
        out = torch.cat((x, c), dim=1)

        out = self.input_layer(out)
        out = self.hidden_layers(out)
        out = self.output_layer(out)

        return out

BATCH_SIZE = 64

dataset = dset.MNIST(
    root="./data/",
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    download=True
)


data_loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    num_workers=32,
    shuffle=True,
    drop_last=True
)

device = torch.device("cuda")

G = Generator(hidden_dim=128).to(device)
D = Discriminator(n_classes=10, hidden_c=128).to(device)

bce_criterion = nn.BCELoss().to(device)
ce_criterion = nn.CrossEntropyLoss().to(device)
D_optimizer = optim.Adam(params=D.parameters(), lr=2e-4)
G_optimizer = optim.Adam(params=G.parameters(), lr=2e-4)

K_D = 1
K_G = 1
epochs = 100

total_batch = len(data_loader)
img_list = []

writer = SummaryWriter("runs/cgan3")
images, labels = next(iter(data_loader))

grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)

images= images.to(device)
# writer.add_graph(D, images)
# writer.add_graph("Generator", G, (torch.randn(BATCH_SIZE, z_dim, 1, 1, device=device), torch.randn(BATCH_SIZE, n_classes, 1, 1, device=device)))

# Train
for epoch in range(epochs):
    avgLoss_D_S_real = 0
    avgLoss_D_S_fake = 0
    avgLoss_D_C_real = 0
    avgLoss_D_C_fake = 0
    avgLoss_G_S = 0
    avgLoss_G_C = 0

    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)

        Y = one_hot(Y, num_classes=n_classes)
        for _ in range(K_D):
            noise = torch.randn(BATCH_SIZE, z_dim, device=device)
            noise = noise.view(BATCH_SIZE, z_dim, 1, 1)

            fake = G(noise, Y.view(-1, n_classes, 1, 1))

            z_real, c_real = D(X)

            real_label = torch.ones((BATCH_SIZE, 1), dtype=torch.float, device=device)
            lossS_real = bce_criterion(z_real, real_label.detach())    
            lossC_real = ce_criterion(c_real, Y.float())

            z_fake, c_fake = D(fake.detach())

            fake_label = torch.zeros((BATCH_SIZE, 1), dtype=torch.float, device=device)
            lossS_fake = bce_criterion(z_fake, fake_label.detach())
            lossC_fake = ce_criterion(c_fake, Y.float())

            lossS = (lossS_real + lossS_fake) / 2
            lossC = (lossC_real + lossC_fake) / 2
            lossD = lossS + lossC

            D.zero_grad()
            lossD.backward()
            D_optimizer.step()

        for _ in range(K_G):
            noise = torch.randn(BATCH_SIZE, z_dim, device=device)
            noise = noise.view(BATCH_SIZE, z_dim, 1, 1)
            fake = G(noise, Y.view(-1, n_classes, 1, 1))
            real_label_G = torch.zeros((BATCH_SIZE, 1), dtype=torch.float, device=device)

            z_fake2, c_fake2 = D(fake)
            lossG_S = bce_criterion(1 - z_fake2, real_label_G.detach())
            lossG_C = ce_criterion(c_fake2, Y.float())
            lossG = lossG_S + 2 * lossG_C

            G.zero_grad()
            lossG.backward()
            G_optimizer.step()

        avgLoss_D_S_real += lossS_real.mean().item() / total_batch
        avgLoss_D_S_fake += lossS_fake.mean().item() / total_batch
        avgLoss_D_C_real += lossC_real.mean().item() / total_batch
        avgLoss_D_C_fake += lossC_fake.mean().item() / total_batch
        avgLoss_G_S += lossG_S.mean().item() / total_batch
        avgLoss_G_C += lossG_C.mean().item() / total_batch

    writer.add_scalar('Loss/avgLoss_D_S_real', avgLoss_D_S_real, epoch+1)
    writer.add_scalar('Loss/avgLoss_D_S_fake', avgLoss_D_S_fake, epoch+1)
    writer.add_scalar('Loss/avgLoss_D_C_real', avgLoss_D_C_real, epoch+1)
    writer.add_scalar('Loss/avgLoss_D_C_fake', avgLoss_D_C_fake, epoch+1)
    writer.add_scalar('Loss/avgLoss_G_S', avgLoss_G_S, epoch+1)
    writer.add_scalar('Loss/avgLoss_G_C', avgLoss_G_C, epoch+1)

    writer.add_image("generated image", generate_example(G), epoch+1)
    print(f"epoch: {epoch+1:3d} avgLoss_D_S_real: {lossS_real:.4f} avgLoss_D_S_fake: {lossS_fake:.4f} avgLoss_D_C_real: {lossC_real:.4f} avgLoss_D_C_fake: {lossC_fake:.4f} avgLoss_G_S: {lossG_S:.4f} avgLoss_G_C: {lossG_C:.4f}")
    # img_list.append(fake)
    generate_example(G)

    if (epoch + 1) % 10 == 0:
        torch.save(G, f"weights/CGAN/CGAN3_G_{epoch+1:03d}.pt")
        torch.save(D, f"weights/CGAN/CGAN3_D_{epoch+1:03d}.pt")

writer.close()


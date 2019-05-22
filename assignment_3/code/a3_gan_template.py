import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

MNIST_SIZE = 784

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, MNIST_SIZE),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.layers(z)        


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.layers = nn.Sequential(
            nn.Linear(MNIST_SIZE, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.layers(img)

    def set_requires_grad(self, flag):
        for param in self.parameters():
            param.requires_grad = flag


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, criterion):
    for epoch in range(args.n_epochs):
        generator_loss = 0
        discriminator_loss = 0
        for i, (imgs, _) in enumerate(dataloader):
            real_label = torch.FloatTensor(1).uniform_(0.7, 1.2)[0]
            fake_label = torch.FloatTensor(1).uniform_(0.0, 0.3)[0]
            
            imgs = imgs.view(-1, MNIST_SIZE)
            rand_sample = torch.randn((imgs.shape[0], args.latent_dim)).cuda()
            labels = torch.zeros(imgs.shape[0], requires_grad=False)

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                rand_sample = rand_sample.cuda()
                labels = labels.cuda()
                
            # Train Discriminator
            optimizer_D.zero_grad()
            discriminator.set_requires_grad(True)
            gen_imgs = generator(rand_sample)
            fake_classification = discriminator(gen_imgs)
            labels.fill_(fake_label)
            loss = criterion(fake_classification, labels)
            discriminator_loss += loss.item()
            loss.backward()
            
            optimizer_D.zero_grad()
            real_classification = discriminator(imgs)
            labels.fill_(real_label)
            loss = criterion(real_classification, labels)
            discriminator_loss += loss.item()
            loss.backward()
            optimizer_D.step()
            # -------------------


            # Train Generator
            optimizer_G.zero_grad()
            discriminator.set_requires_grad(False)
            gen_imgs = generator(rand_sample)
            fake_classification = discriminator(gen_imgs)
            loss = criterion(fake_classification, labels)
            generator_loss += loss.item()
            loss.backward()
            optimizer_G.step()
            # ---------------


            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(gen_imgs[:25],
                           'images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)
        print(f"Epoch: {epoch}, \
               discriminator loss: {(discriminator_loss / len(dataloader)):.3f}, \
               generator loss: {(generator_loss / len(dataloader)):.3f}")


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, ),(0.5, ))
                           ])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(args.latent_dim)
    discriminator = Discriminator()

    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, criterion)

    # You can save your generator here to re-use it to generate images for your
    torch.save(generator, "GAN-model.pt")
    return generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()

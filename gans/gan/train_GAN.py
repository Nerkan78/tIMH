
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
# from code.gans.gan.gan_trainer_bce import gan_trainer
from gans.gan.dcgan import Generator, Discriminator


from tqdm import tqdm

def gan_trainer(dataloader, net_D, net_G, device, lr_D=2e-4, lr_G=2e-4,
                dim_z=100,  epoch_number=50,
                verbose=5, path_f='.'):

    fixed_noise = torch.randn(64, dim_z, 1, 1, device=device)

    real_label = 1
    fake_label = 0

    optimizer_D = optim.Adam(net_D.parameters(), lr=lr_D)

    optimizer_G = optim.Adam(net_G.parameters(), lr=lr_G)
    criterion = nn.BCELoss()

    print("Start")
    for epoch in tqdm(range(epoch_number)):
        for i, data in enumerate(dataloader, 0):
            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            net_D.zero_grad()
            real = data[0].to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, device=device)
            output = net_D(real).view(-1)
            loss_D_real = criterion(output.to(torch.float32), label.to(torch.float32))
            loss_D_real.backward()

            # stats
            D_x = output.mean().item()

            noise = torch.randn(b_size, dim_z, 1, 1, device=device)
            fake = net_G(noise)
            label.fill_(fake_label)
            output = net_D(fake.detach()).view(-1)
            loss_D_fake = criterion(output.to(torch.float32), label.to(torch.float32))
            loss_D_fake.backward()
            optimizer_D.step()

            # stats
            loss_D = loss_D_real.item() + loss_D_fake.item()
            D_G_z1 = output.mean().item()

            # Update G network: maximize log(D(G(z)))
            net_G.zero_grad()
            label.fill_(real_label)
            output = net_D(fake).view(-1)
            loss_G = criterion(output.to(torch.float32), label.to(torch.float32))
            loss_G.backward()
            optimizer_G.step()

            # stats
            D_G_z2 = output.mean().item()



            if (i % (10 * verbose) == 0):
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, epoch_number, i, len(dataloader),
                         loss_D, loss_G.item(), D_x, D_G_z1, D_G_z2))

        if (epoch % verbose == 0):
            torch.save(net_G.state_dict(), '%s/net_G_epoch_%d.pth' % (path_f, epoch))
            torch.save(net_D.state_dict(), '%s/net_D_epoch_%d.pth' % (path_f, epoch))

            fake = net_G(fixed_noise)
            vutils.save_image(fake.detach(), '%s/img_grid_G/net_G_samples_epoch_%03d.png' % (path_f, epoch),
                              normalize=True)
    return 0

def train(nc, ndf, device, dim_z, ngf, net_D_pf, net_G_pf, image_size, batch_size,
          lr_D=2e-4, lr_G=2e-4, epoch_number=50,
          verbose=5, path_f='.',):

    net_D = Discriminator(nc, ndf).to(device)
    if net_D_pf is not None:
        net_D.load_state_dict(torch.load(net_D_pf, map_location=device))

    net_G = Generator(nc, dim_z, ngf).to(device)
    if net_G_pf is not None:
        net_G.load_state_dict(torch.load(net_G_pf, map_location=device))

    dataset = dset.CIFAR10(root='./data', download=True,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

    gan_trainer(dataloader=dataloader, net_D=net_D, net_G=net_G,
                device=device, lr_D=lr_D, lr_G=lr_G,
                dim_z=dim_z, epoch_number=epoch_number,
                verbose=verbose, path_f=path_f)

    return net_D, net_G


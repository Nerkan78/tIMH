import os

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from gans.gan.dcgan import Generator, Discriminator, WrapD


def train_bce(D, G, dataloader, lr, epoch_n, device, verbose, ml_ls):
    read_names = []
    loss_history = []
    ar_history = []
    dim_z = 100
    real_label = 1
    fake_label = 0
    D.to(device)
    G.to(device).eval()

    optimizer = optim.Adam(D.parameters(), lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=ml_ls, gamma=0.1)
    criterion = nn.BCELoss()
    count_object = 0
    print('Start')
    for epoch in range(epoch_n):
        for i, data in (enumerate(dataloader)):
            x_s = data[0].to(device)
            b_size = x_s.size(0)
            count_object += b_size

            with torch.no_grad():
                noise = torch.randn(b_size, dim_z, 1, 1, device=device)
                x_p = G(noise)
                batch = torch.cat([x_s, x_p])
                l_s = torch.full((b_size, ), real_label, device=device)
                l_p = torch.full((b_size, ), fake_label, device=device)
                # label = torch.cat([l_s, l_p])
                #
                # idx = torch.randperm(2 * b_size)
                # label = label[idx]
                # batch = batch[idx]

            fake_output = D(x_p.detach()).view(-1)
            real_output = D(x_s.detach()).view(-1)
            real_loss = criterion(real_output.to(torch.float32), l_s.to(torch.float32))
            fake_loss = criterion(fake_output.to(torch.float32), l_p.to(torch.float32))
            loss = real_loss + fake_loss

            loss_history.append([count_object, loss.item()])

            optimizer.zero_grad()
            real_loss.backward()
            fake_loss.backward()

            optimizer.step()
            scheduler.step()

            if i % verbose == 0:
                with torch.no_grad():
                    fake_score = torch.mean(D(x_p)).item()
                    true_score = torch.mean(D(x_s)).item()
                    ar = torch.mean(D.ar(x_s, x_p)).item()
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tFake score: %4f\tReal score: %4f, AR: %4f'
                      % (epoch, epoch_n-1, i, len(dataloader), loss.item(), fake_score, true_score, ar))

    torch.cuda.empty_cache()
    return D


def train(net_G_pf, device, nc=3, dim_z=100, ngf=64, ndf=64, image_size=32, batch_size=128,
          lr=2e-4, lr_decay = [30, 50], epoch_number=50,
          verbose=5):

    G = Generator( nc, dim_z, ngf)
    G.load_state_dict(torch.load(net_G_pf, map_location='cpu'))
    G.eval()
    D = Discriminator(nc, ndf)

    W = WrapD(D)

    dataset = dset.CIFAR10(root='./data', download=True,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device(device)


    path_f = 'gG_tD_' + 'cifar'
    if not os.path.exists(path_f):
        os.makedirs(path_f)


    D = train_bce(D=W, G=G, dataloader=dataloader,
              lr=lr, epoch_n=epoch_number, device=device,
              verbose= verbose, ml_ls=lr_decay)
    return G, D



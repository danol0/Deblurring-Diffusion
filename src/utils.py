from src.model import CNN, DeblurringDiffusion, UNet
import torch
import os
from torchvision.utils import save_image, make_grid
import numpy as np
from torchvision import transforms
import torch.nn as nn


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def define_model(opt):
    if opt.net == "CNN":
        net = CNN(
            in_channels=1,
            expected_shape=(28, 28),
            n_hidden=opt.n_hidden,
            act=nn.GELU,
        )
    elif opt.net == "UNet":
        net = UNet()
    else:
        raise ValueError(f"Unknown network: {opt.net}")

    model = DeblurringDiffusion(
        net=net, n_T=opt.n_T, scheduler=opt.scheduler, deg=opt.deg_type
    )

    return model


def save_model(name, savedir, model, optim, opt, losses):
    name += ".pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "opt": opt,
            "losses": losses,
        },
        os.path.join(mkdir(os.path.join(savedir, "checkpoints")), name),
    )


def sample_grid(name, model, savedir, n, device):
    name += ".png"
    with torch.no_grad():
        xh = model.sample(n, (1, 28, 28), device=device)
        xh = (xh + 1) / 2.0
        save_image(
            make_grid(xh, nrow=np.sqrt(n).astype(int)),
            os.path.join(mkdir(os.path.join(savedir, "samples")), name),
        )


def get_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def batch2rgb(x):
    return x.repeat(1, 3, 1, 1)

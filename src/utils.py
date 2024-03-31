from src.model import CNN, DeblurringDiffusion, UNet
import torch
import os
from torchvision.utils import save_image, make_grid
import numpy as np
from torchvision import transforms
import torch.nn as nn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator


def mkdir(path):
    """Creates a directory if it does not exist."""

    if not os.path.exists(path):
        os.makedirs(path)
    return path


def define_model(opt):
    """Defines the model architecture based on the command line options."""

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

    if opt.deg_type == "noise":
        model = DDPM(
            net=net, betas=(opt.beta1, opt.betaT), n_T=opt.n_T, scheduler=opt.scheduler
        )
    elif "blur" in opt.deg_type:
        model = DeblurringDiffusion(
            net=net, n_T=opt.n_T, scheduler=opt.scheduler, deg=opt.deg_type
        )
    else:
        raise ValueError(f"Unknown deg type: {opt.deg_type}")

    return model


def save_model(name, savedir, model, optim, opt, losses):
    """Saves the model, optimizer, options, and losses to a checkpoint."""

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


def load_model(ckpt_dir):
    """Loads a model from a checkpoint."""

    ckpt = torch.load(ckpt_dir)
    model = define_model(ckpt["opt"])
    model.load_state_dict(ckpt["model"])
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    model.eval()
    return model, accelerator.device


def sample_grid(name, model, savedir, n, device):
    """Samples a grid of images from the model and saves it to a file."""

    name += ".png"
    with torch.no_grad():
        xh = model.sample(n, (1, 28, 28), device=device)
        xh = (xh + 1) / 2.0
        save_image(
            make_grid(xh, nrow=np.sqrt(n).astype(int)),
            os.path.join(mkdir(os.path.join(savedir, "samples")), name),
        )


def get_transforms():
    """Transforms used for the MNIST dataset."""

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def batch2rgb(x):
    """Converts a batch of grayscale images to RGB."""

    return x.repeat(1, 3, 1, 1)


@torch.no_grad()
def get_fids(ckpt_dirs, n=200, fid=None):
    """Computes FID scored for a list of checkpoints"""

    if n > 500:
        print("Warning: Consider using batches")
    output = []
    if fid is None:
        fid = FrechetInceptionDistance(normalize=True, reset_real_features=False)
        dataloader = DataLoader(
            MNIST(
                "./data", train=False, download=True, transform=transforms.ToTensor()
            ),
            batch_size=n,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )
        real = next(iter(dataloader))[0]
        real = batch2rgb(real)
        fid.update(real, real=True)

    for ckpt_dir in tqdm(ckpt_dirs):
        fid.reset()
        # Load Model
        ckpt = torch.load(ckpt_dir)
        model = define_model(ckpt["opt"])
        model.load_state_dict(ckpt["model"])
        model.eval()
        accelerator = Accelerator()
        model = accelerator.prepare(model)
        # Sample
        fake = model.sample(n, (1, 28, 28), device=accelerator.device)
        fake = (fake + 1) / 2.0
        fake = torch.clamp(fake, 0, 1)
        fake = batch2rgb(fake)
        fake = fake.cpu()
        fid.update(fake, real=False)

        score = fid.compute().item()
        print(f"FID for {ckpt_dir}: {score}")
        output.append(score)

    return output

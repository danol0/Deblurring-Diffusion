import torch
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import wandb
import numpy as np
import os

from options import parse_args
from src.utils import define_model, save_model, mkdir, sample_grid, get_transforms


if __name__ == "__main__":

    # Initialise
    opt, str_opt = parse_args()
    print(str_opt)
    savedir = mkdir(os.path.join("./runs", opt.deg_type, opt.name))
    with open(os.path.join(savedir, "opt.txt"), "wt") as opt_file:
        opt_file.write(str_opt)
        opt_file.write("\n")

    # os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(project="Diffusion", config=opt, name=opt.name)

    # Model & dataset
    model = define_model(opt)
    optim = torch.optim.Adam(model.parameters(), lr=opt.lr)

    if opt.use_ckpt:
        ckpt_dir = os.path.join(
            "./runs", opt.deg_type, opt.name, f"checkpoints/epoch_{opt.use_ckpt}.pt"
        )
        ckpt = torch.load(ckpt_dir)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        print(f"Loaded checkpoint: {ckpt_dir}")

    tf = get_transforms()
    dataloader = DataLoader(
        MNIST("./data", train=True, download=True, transform=tf),
        batch_size=128,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    testloader = DataLoader(
        MNIST("./data", train=False, download=True, transform=tf),
        batch_size=128,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    # Accelerator
    accelerator = Accelerator()
    model, optim, dataloader, testloader = accelerator.prepare(
        model, optim, dataloader, testloader
    )
    wandb.config.update({"device": accelerator.device})
    print(f"Using device: {accelerator.device}")

    # Training loop
    print(f"Total gradient steps: {len(dataloader) * opt.n_epoch}")
    losses = (
        ckpt["losses"]
        if opt.use_ckpt
        else {"train": [], "test_avg": [], "train_avg": []}
    )
    pbar = tqdm(total=len(dataloader), position=0, leave=True)

    for epoch in range(1, opt.n_epoch + 1):
        epoch += opt.use_ckpt if opt.use_ckpt else 0
        pbar.reset()
        model.train()

        for x, _ in dataloader:
            optim.zero_grad()

            loss = model(x)

            accelerator.backward(loss)

            losses["train"].append(loss.item())
            avg_loss = np.average(losses["train"][-100:])
            pbar.set_description(f"Epoch: {epoch}, Loss: {avg_loss:.3g}")
            pbar.update(1)

            optim.step()

        model.eval()
        epoch_average_loss = np.average(losses["train"][-len(dataloader) :])
        wandb.log({"loss": epoch_average_loss})
        losses["train_avg"].append(epoch_average_loss)

        # Test loop
        with torch.no_grad():
            test_losses = []
            for x, _ in testloader:
                loss = model(x)
                test_losses.append(loss.item())
            test_average_loss = np.average(test_losses)
            losses["test_avg"].append(test_average_loss)
            wandb.log({"test_loss": test_average_loss})

        if epoch % 5 == 0 or epoch == 1:  # Save samples and model every 5 epochs
            sample_grid(f"epoch_{epoch}", model, savedir, 25, accelerator.device)
            save_model(f"epoch_{epoch}", savedir, model, optim, opt, losses)

    # Save final sample and weights
    sample_grid(f"epoch_{opt.n_epoch}", model, savedir, 64, accelerator.device)
    save_model(f"epoch_{opt.n_epoch}", savedir, model, optim, opt, losses)

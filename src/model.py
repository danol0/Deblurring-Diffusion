import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as tvF
from typing import Tuple

# --- Diffusion ---


def blur_schedules(
    n_T: int, scheduler: str = "linear", beta1: float = 1e-4, betaT: float = 0.02
) -> list:
    """Generate the standard deviation schedule for the blur kernel.

    Args:
        n_T: Number of diffusion steps.
        scheduler: Type of scheduler to use, either "constant", "linear" or "exp".
        beta1, betaT: Parameters that define schedules.

    Returns:
        List of standard deviations for each time step.
    """

    if scheduler == "constant":
        return [1 + betaT] * n_T
    elif scheduler == "linear":
        return [(betaT) * t + beta1 for t in range(n_T)]
    elif scheduler == "exp":
        # Currently cannot be parameterized for ease of use
        return [0.5 * np.exp(0.03 * t) for t in range(n_T)]
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")


class DeblurringDiffusion(nn.Module):
    def __init__(
        self,
        net: nn.Module,
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
        scheduler: str = "linear",
        deg: str = "blur",
    ) -> None:
        """
        Diffusion model that inverts Gaussian blurs.

        Args:
            net: Deep learning model used to predict the original image.
            n_T: Number of diffusion steps.
            criterion: Loss function to use for training.
            scheduler: Schedule for the standard deviation of the blur kernel.
            deg: Type of degradation to use, either "blur" or "blurfade".
        """
        super().__init__()

        self.net = net
        self.n_T = n_T
        self.criterion = criterion

        if deg == "blur":
            self.deg = self.iterative_blur
            self.shades = (-0.85, -0.65)
        elif deg == "blurfade":
            self.deg = self.blur_and_fade
            # Pre-computed beta_t for a 100 step diffusion
            if n_T != 100:
                print("Consider changing the beta_t schedule")
            # TODO: allow this to be parameterized
            t = torch.linspace(0, 1, n_T)
            self.register_buffer("beta_t", 1e-3 * (0.2 / 1e-3) ** t)
            self.shades = (-1, 1)
        else:
            raise ValueError(f"Unknown degradation type: {deg}")

        self.stds = blur_schedules(n_T, scheduler)

    def iterative_blur(
        self,
        x: torch.Tensor,
        t: int,
        start: int = 0,
        col: torch.Tensor = None,  # Dummy argument for compatibility
    ) -> torch.Tensor:
        """Iteratively blur a batch of images.

        Args:
            x: Batch of images to blur.
            t: Time step to blur to.
            start: Time step to start from.
            col: Dummy argument for compatibility, ignored.

        Returns:
            Degraded batch of images.
        """
        for i in range(start, t):
            x = tvF.gaussian_blur(x, 7, self.stds[i])
        return x

    def blur_and_fade(
        self,
        x: torch.Tensor,
        t: int,
        start: int = 0,
        col: torch.Tensor = None,
    ) -> torch.Tensor:
        """Iteratively fade and blur a batch of images.

        Args:
            x: Batch of images to blur.
            t: Time step to degrade to.
            start: Time step to start from.
            col: Colour to fade to, if None a random colour is chosen.

        Returns:
            Degraded batch of images.
        """
        if col is None:
            col = torch.rand(x.shape[0], 1, 1, 1, device=x.device) * 2 - 1
        for i in range(start, t):
            beta_t = self.beta_t[i, None, None, None].to(x.device)
            x = torch.sqrt(beta_t) * col + torch.sqrt(1 - beta_t) * x
            x = tvF.gaussian_blur(x, 7, self.stds[i])
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training step for the model."""

        # Single time step each time - easier to parallelize
        t = torch.randint(1, self.n_T + 1, (1,), device=x.device).expand(x.shape[0])

        z_t = self.deg(x, t[0].item())

        # Predict the original image
        return self.criterion(x, self.net(z_t, t / self.n_T))

    @torch.no_grad()
    def sample(
        self,
        n_sample: int,
        size: Tuple,
        device: torch.device,
        std: float = 0.002,
        skip: int = None,
        naive: bool = False,
        show_steps: int = None,
        z_T: torch.Tensor = None,
        shades: Tuple = None,
    ) -> torch.Tensor:
        """Algorithms 1 (naive) & 2 (default) in Bansal et al. 2022

        Args:
            n_sample: Number of samples to generate
            size: Image size (e.g. (1, 32, 32))
            std: Standard deviation of the noise added to z_T
            skip: Highest t step to re-blur to, if None the full diffusion is run
            naive: Use algorithm 1 in Bansal
            show_steps: Number of intermediate steps to show, if None only the z_0 is returned
            z_T: Optional precomputed z_T to start from, if None a random z_T is generated
            shades: Lower and upper bounds for shade of z_T, if None the defaults for the deg type are used

        Returns:
            Batch of generated samples, including intermediate steps if requested
        """
        # Define z_T
        lower, upper = self.shades if shades is None else sorted(shades)
        if z_T is not None:
            assert z_T.shape[0] == n_sample, "z_T must have the same batch size"
            assert z_T.shape[1:] == size, "z_T must have the same shape as size"
            z_t = z_T.to(device)
        else:
            shade = lower + (upper - lower) * torch.rand(
                n_sample, 1, 1, 1, device=device
            )
            z_t = shade + torch.randn(n_sample, *size, device=device) * std

        _one = torch.ones(n_sample, device=device)

        # Handle showing steps and skipping
        t_start = skip if skip else self.n_T
        _cuts = None
        if show_steps:
            _cuts = np.linspace(0, t_start, show_steps, dtype=int)[1:-1]
            steps = z_t.clone()

        if skip:  # Do first step outside loop if skipping
            z_0 = self.net(z_t, _one)
            z_t = self.deg(z_0, t_start)
            if show_steps:
                steps = torch.cat((steps, z_t.clone()), dim=0)
                _cuts = _cuts[:-1]

        # Sampling loop
        for t in range(t_start, 0, -1):
            z_0 = self.net(z_t, (t / self.n_T) * _one)

            if t > 1:
                if naive:
                    z_t = self.deg(z_0, t - 1)
                else:
                    # Make sure the shade is the same for both degradation steps
                    shade = lower + (upper - lower) * torch.rand(
                        n_sample, 1, 1, 1, device=device
                    )
                    z_t_sub_1 = self.deg(z_0, t - 1, col=shade)
                    z_t += -self.deg(z_t_sub_1, t, start=t - 1, col=shade) + z_t_sub_1

            if show_steps and t in _cuts:
                steps = torch.cat((steps, z_t.clone()), dim=0)

        if show_steps:
            steps = torch.cat((steps, z_0.clone()), dim=0)

        return steps if show_steps else z_0

    @torch.no_grad()
    def encode_decode(
        self, x: torch.Tensor, t: int, device: torch.device, show_steps: int = 3
    ) -> torch.Tensor:
        """Conditional sampling with intermediate steps.

        Args:
            x: Batch of images to encode and decode.
            t: Max time step to decode to.
            device: Device to run on.
            show_steps: Number of intermediate steps to show.

        Returns:
            Batched tensor of images and intermediate steps.
        """

        _cuts = np.linspace(0, t, show_steps, dtype=int)[1:]
        x = x.to(device)
        x = x.unsqueeze(0) if len(x.shape) == 3 else x
        steps = x.clone()

        # Choose a colour for fading - this will be ignored for only blur
        col = torch.rand(x.shape[0], 1, 1, 1, device=device) * 2 - 1

        for i in range(1, t):
            x = self.deg(x, i, start=i - 1, col=col)
            if i in _cuts:
                steps = torch.cat((steps, x.clone()), dim=0)

        for i in range(t, 0, -1):
            z_0 = self.net(x, (i / self.n_T) * torch.ones(1, device=device))

            if i > 1:
                z_t_sub_1 = self.deg(z_0, i - 1, col=col)
                x = x - self.deg(z_t_sub_1, i, start=i - 1, col=col) + z_t_sub_1
            else:
                x = z_0

            if i in _cuts:
                steps = torch.cat((steps, x.clone()), dim=0)

        steps = torch.cat((steps, x.clone()), dim=0)
        return steps


# --- Convolutional models ---


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        dim1: int = 32,
        time_embeddings: int = 16,
        act: nn.Module = nn.GELU,
    ):
        """Simple UNet model.

        Args:
            in_channels: Number of input channels.
            dim1: Number of channels in the first convolutional layer.
            time_embeddings: Number of time embeddings to use.
            act: Activation function to use."""

        super().__init__()

        # Encoder
        self.conv1 = ConvNextBlock(
            in_channels, dim1, time_dim=dim1, expected_shape=(28, 28)
        )
        self.maxpool1 = self.maxpool()
        self.conv2 = ConvNextBlock(
            dim1, dim1 * 2, time_dim=dim1, expected_shape=(14, 14)
        )
        self.maxpool2 = self.maxpool()

        self.middle = ConvNextBlock(
            dim1 * 2, dim1 * 4, time_dim=dim1, expected_shape=(7, 7)
        )

        # Decoder
        self.upsample2 = self.transposed_block(dim1 * 4, dim1 * 2)
        self.upconv2 = ConvNextBlock(
            dim1 * 4, dim1 * 2, time_dim=dim1, expected_shape=(14, 14)
        )
        self.upsample1 = self.transposed_block(dim1 * 2, dim1)
        self.upconv1 = ConvNextBlock(
            dim1 * 2, dim1, time_dim=dim1, expected_shape=(28, 28)
        )

        self.final = nn.Conv2d(dim1, in_channels, kernel_size=1, stride=1, padding=0)

        self.time_embed = nn.Sequential(
            nn.Linear(time_embeddings * 2, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, dim1),
        )
        frequencies = torch.tensor(
            [0] + [2 * np.pi * 1.5**i for i in range(time_embeddings - 1)]
        )
        self.register_buffer("frequencies", frequencies)

    def time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        phases = torch.concat(
            (
                torch.sin(t[:, None] * self.frequencies[None, :]),
                torch.cos(t[:, None] * self.frequencies[None, :]) - 1,
            ),
            dim=1,
        )

        return self.time_embed(phases)

    def transposed_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
    ) -> nn.Module:
        return nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )

    def maxpool(
        self,
        dropout_rate: float = 0.5,
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 0,
    ) -> nn.Module:
        return nn.Sequential(
            nn.MaxPool2d(kernel_size, stride, padding), nn.Dropout2d(dropout_rate)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = self.time_encoding(t)

        # Encode
        x1 = self.conv1(x, t)  # 28x28
        x2 = self.maxpool1(x1)  # 14x14
        x2 = self.conv2(x2, t)
        out = self.maxpool2(x2)  # 7x7
        out = self.middle(out, t)

        # Decode
        out = self.upsample2(out)
        out = torch.cat([out, x2], dim=1)
        out = self.upconv2(out, t)
        out = self.upsample1(out)
        out = torch.cat([out, x1], dim=1)
        out = self.upconv1(out, t)

        out = self.final(out)

        return out


class ConvNextBlock(nn.Module):
    """Source: Cold Diffusion / https://arxiv.org/pdf/2201.03545.pdf"""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        time_dim: int = 32,
        mult: int = 2,
        norm: bool = True,
        expected_shape: Tuple = (28, 28),
    ) -> None:
        super().__init__()

        self.time_mlp = nn.Sequential(nn.GELU(), nn.Linear(time_dim, dim))

        self.depthwise = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.LayerNorm((dim, *expected_shape)) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.residual = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.depthwise(x)
        # Add time embedding
        t_emb = self.time_mlp(t)
        h += t_emb.unsqueeze(-1).unsqueeze(-1)
        h = self.net(h)
        return h + self.residual(x)


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        expected_shape: Tuple,
        act: nn.Module = nn.GELU,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.LayerNorm((out_channels, *expected_shape)),
            act(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        expected_shape: Tuple = (28, 28),
        n_hidden: Tuple = (64, 128, 64),
        kernel_size: int = 7,
        last_kernel_size: int = 3,
        time_embeddings: int = 16,
        act: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()
        last = in_channels

        self.blocks = nn.ModuleList()
        for hidden in n_hidden:
            self.blocks.append(
                CNNBlock(
                    last,
                    hidden,
                    expected_shape=expected_shape,
                    kernel_size=kernel_size,
                    act=act,
                )
            )
            last = hidden

        self.blocks.append(
            nn.Conv2d(
                last,
                in_channels,
                last_kernel_size,
                padding=last_kernel_size // 2,
            )
        )

        self.time_embed = nn.Sequential(
            nn.Linear(time_embeddings * 2, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, n_hidden[0]),
        )
        frequencies = torch.tensor(
            [0] + [2 * np.pi * 1.5**i for i in range(time_embeddings - 1)]
        )
        self.register_buffer("frequencies", frequencies)

    def time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        phases = torch.concat(
            (
                torch.sin(t[:, None] * self.frequencies[None, :]),
                torch.cos(t[:, None] * self.frequencies[None, :]) - 1,
            ),
            dim=1,
        )

        return self.time_embed(phases)[:, :, None, None]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        embed = self.blocks[0](x)

        embed += self.time_encoding(t)

        for block in self.blocks[1:]:
            embed = block(embed)

        return embed

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _NullAutocast:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


try:
    _modern_autocast = torch.amp.autocast

    def autocast(enabled: bool) -> Any:
        return _modern_autocast(device_type="cuda", enabled=enabled)

except Exception:  # pragma: no cover - older torch fallback
    try:
        _legacy_autocast = torch.cuda.amp.autocast

        def autocast(enabled: bool) -> Any:
            return _legacy_autocast(enabled=enabled)

    except Exception:  # pragma: no cover - no autocast available
        def autocast(enabled: bool) -> Any:
            return _NullAutocast(enabled=enabled)


def _meshgrid_ij(y: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return torch.meshgrid(y, x, indexing="ij")
    except TypeError:  # pragma: no cover - old torch fallback
        return torch.meshgrid(y, x)


class InputPadder:
    """Pads images such that dimensions are divisible by 8."""

    def __init__(self, dims: torch.Size | tuple[int, ...], mode: str = "sintel") -> None:
        self.ht, self.wd = int(dims[-2]), int(dims[-1])
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8

        if mode == "sintel":
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs: torch.Tensor) -> list[torch.Tensor]:
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x: torch.Tensor) -> torch.Tensor:
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


def bilinear_sampler(
    img: torch.Tensor,
    coords: torch.Tensor,
    mode: str = "bilinear",
    mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Wrapper for grid_sample that uses pixel coordinates."""

    h, w = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2.0 * xgrid / max(w - 1, 1) - 1.0
    ygrid = 2.0 * ygrid / max(h - 1, 1) - 1.0

    grid = torch.cat([xgrid, ygrid], dim=-1)
    sampled = F.grid_sample(img, grid, mode=mode, align_corners=True)

    if not mask:
        return sampled

    valid = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
    return sampled, valid.float()


def coords_grid(batch: int, ht: int, wd: int, device: torch.device | str) -> torch.Tensor:
    ys = torch.arange(ht, device=device)
    xs = torch.arange(wd, device=device)
    yy, xx = _meshgrid_ij(ys, xs)
    coords = torch.stack([xx, yy], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8.0 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, norm_fn: str = "group", stride: int = 1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8
        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes) if stride != 1 else nn.Sequential()
        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            self.norm3 = nn.BatchNorm2d(planes) if stride != 1 else nn.Sequential()
        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            self.norm3 = nn.InstanceNorm2d(planes) if stride != 1 else nn.Sequential()
        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
        else:
            raise ValueError(f"Unsupported norm_fn: {norm_fn}")

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm3,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, norm_fn: str = "group", stride: int = 1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8
        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes) if stride != 1 else nn.Sequential()
        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes // 4)
            self.norm2 = nn.BatchNorm2d(planes // 4)
            self.norm3 = nn.BatchNorm2d(planes)
            self.norm4 = nn.BatchNorm2d(planes) if stride != 1 else nn.Sequential()
        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes // 4)
            self.norm2 = nn.InstanceNorm2d(planes // 4)
            self.norm3 = nn.InstanceNorm2d(planes)
            self.norm4 = nn.InstanceNorm2d(planes) if stride != 1 else nn.Sequential()
        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            self.norm4 = nn.Sequential()
        else:
            raise ValueError(f"Unsupported norm_fn: {norm_fn}")

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm4,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, output_dim: int = 128, norm_fn: str = "batch", dropout: float = 0.0) -> None:
        super().__init__()

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)
        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64)
        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
        else:
            raise ValueError(f"Unsupported norm_fn: {norm_fn}")

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1, norm_fn=norm_fn)
        self.layer2 = self._make_layer(96, stride=2, norm_fn=norm_fn)
        self.layer3 = self._make_layer(128, stride=2, norm_fn=norm_fn)

        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim: int, stride: int, norm_fn: str) -> nn.Sequential:
        layer1 = ResidualBlock(self.in_planes, dim, norm_fn=norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, norm_fn=norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        is_list = isinstance(x, (tuple, list))
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class SmallEncoder(nn.Module):
    def __init__(self, output_dim: int = 128, norm_fn: str = "batch", dropout: float = 0.0) -> None:
        super().__init__()

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(32)
        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(32)
        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
        else:
            raise ValueError(f"Unsupported norm_fn: {norm_fn}")

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1, norm_fn=norm_fn)
        self.layer2 = self._make_layer(64, stride=2, norm_fn=norm_fn)
        self.layer3 = self._make_layer(96, stride=2, norm_fn=norm_fn)

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim: int, stride: int, norm_fn: str) -> nn.Sequential:
        layer1 = BottleneckBlock(self.in_planes, dim, norm_fn=norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, norm_fn=norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        is_list = isinstance(x, (tuple, list))
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


try:
    import alt_cuda_corr  # type: ignore
except Exception:  # pragma: no cover - optional extension
    alt_cuda_corr = None


class CorrBlock:
    def __init__(self, fmap1: torch.Tensor, fmap2: torch.Tensor, num_levels: int = 4, radius: int = 4) -> None:
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid: list[torch.Tensor] = []

        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for _ in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid: list[torch.Tensor] = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            yy, xx = _meshgrid_ij(dy, dx)
            delta = torch.stack([xx, yy], dim=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / (2**i)
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr_sampled = bilinear_sampler(corr, coords_lvl)
            corr_sampled = corr_sampled.view(batch, h1, w1, -1)
            out_pyramid.append(corr_sampled)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        scale = torch.sqrt(torch.tensor(float(dim), device=fmap1.device))
        return corr / scale


class AltCudaCorr(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        fmap1: torch.Tensor,
        fmap2_i: torch.Tensor,
        coords: torch.Tensor,
        r: int,
    ) -> tuple[torch.Tensor]:
        if alt_cuda_corr is None:
            raise RuntimeError(
                "alternate_corr requested but alt_cuda_corr extension is not available"
            )

        ctx.save_for_backward(fmap1, fmap2_i, coords)
        ctx.r = r
        corr, = alt_cuda_corr.forward(fmap1, fmap2_i, coords, r)
        return (corr,)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        corr_grad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        if alt_cuda_corr is None:
            raise RuntimeError(
                "alternate_corr requested but alt_cuda_corr extension is not available"
            )

        fmap1, fmap2_i, coords = ctx.saved_tensors
        corr_grad = corr_grad.contiguous()
        fmap1_grad, fmap2_grad, coords_grad = alt_cuda_corr.backward(
            fmap1,
            fmap2_i,
            coords,
            corr_grad,
            ctx.r,
        )
        return fmap1_grad, fmap2_grad, coords_grad, None


class AlternateCorrBlock:
    def __init__(self, fmap1: torch.Tensor, fmap2: torch.Tensor, num_levels: int = 4, radius: int = 4) -> None:
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for _ in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.permute(0, 2, 3, 1)
        batch, h, w, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list: list[torch.Tensor] = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / (2**i)).reshape(batch, 1, h, w, 2).contiguous()
            corr, = AltCudaCorr.apply(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(batch, -1, h, w)
        scale = torch.sqrt(torch.tensor(float(dim), device=corr.device))
        return corr / scale


class FlowHead(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim: int = 128, input_dim: int = 192 + 128) -> None:
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim: int = 128, input_dim: int = 192 + 128) -> None:
        super().__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class SmallMotionEncoder(nn.Module):
    def __init__(self, args: object) -> None:
        super().__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicMotionEncoder(nn.Module):
    def __init__(self, args: object) -> None:
        super().__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SmallUpdateBlock(nn.Module):
    def __init__(self, args: object, hidden_dim: int = 96) -> None:
        super().__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82 + 64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(
        self,
        net: torch.Tensor,
        inp: torch.Tensor,
        corr: torch.Tensor,
        flow: torch.Tensor,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        return net, None, delta_flow


class BasicUpdateBlock(nn.Module):
    def __init__(self, args: object, hidden_dim: int = 128, input_dim: int = 128) -> None:
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=input_dim + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

    def forward(
        self,
        net: torch.Tensor,
        inp: torch.Tensor,
        corr: torch.Tensor,
        flow: torch.Tensor,
        upsample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class RAFT(nn.Module):
    def __init__(self, args: object) -> None:
        super().__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if "dropout" not in self.args:
            self.args.dropout = 0

        if "alternate_corr" not in self.args:
            self.args.alternate_corr = False

        if "mixed_precision" not in self.args:
            self.args.mixed_precision = False

        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn="instance", dropout=args.dropout)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn="none", dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn="instance", dropout=args.dropout)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn="batch", dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def initialize_flow(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, _, h, w = img.shape
        coords0 = coords_grid(n, h // 8, w // 8, device=img.device)
        coords1 = coords_grid(n, h // 8, w // 8, device=img.device)
        return coords0, coords1

    def upsample_flow(self, flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n, _, h, w = flow.shape
        mask = mask.view(n, 1, 9, 8, 8, h, w)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(n, 2, 9, 1, 1, h, w)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(n, 2, 8 * h, 8 * w)

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: int = 12,
        flow_init: torch.Tensor | None = None,
        upsample: bool = True,
        test_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor]:
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions: list[torch.Tensor] = []
        for _ in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            coords1 = coords1 + delta_flow

            if upsample and up_mask is not None:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            else:
                flow_up = upflow8(coords1 - coords0)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions


# Flow visualization adapted from Tom Runia's OpticalFlow_Visualization (MIT).
def make_colorwheel() -> np.ndarray:
    ry = 15
    yg = 6
    gc = 4
    cb = 11
    bm = 13
    mr = 6

    ncols = ry + yg + gc + cb + bm + mr
    colorwheel = np.zeros((ncols, 3), dtype=np.float32)
    col = 0

    colorwheel[0:ry, 0] = 255
    colorwheel[0:ry, 1] = np.floor(255 * np.arange(0, ry) / ry)
    col += ry

    colorwheel[col : col + yg, 0] = 255 - np.floor(255 * np.arange(0, yg) / yg)
    colorwheel[col : col + yg, 1] = 255
    col += yg

    colorwheel[col : col + gc, 1] = 255
    colorwheel[col : col + gc, 2] = np.floor(255 * np.arange(0, gc) / gc)
    col += gc

    colorwheel[col : col + cb, 1] = 255 - np.floor(255 * np.arange(cb) / cb)
    colorwheel[col : col + cb, 2] = 255
    col += cb

    colorwheel[col : col + bm, 2] = 255
    colorwheel[col : col + bm, 0] = np.floor(255 * np.arange(0, bm) / bm)
    col += bm

    colorwheel[col : col + mr, 2] = 255 - np.floor(255 * np.arange(mr) / mr)
    colorwheel[col : col + mr, 0] = 255

    return colorwheel


def flow_uv_to_colors(u: np.ndarray, v: np.ndarray, convert_to_bgr: bool = False) -> np.ndarray:
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1.0) / 2.0 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75

        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)

    return flow_image


def flow_to_image(flow_uv: np.ndarray, clip_flow: float | None = None, convert_to_bgr: bool = False) -> np.ndarray:
    if flow_uv.ndim != 3 or flow_uv.shape[2] != 2:
        raise ValueError("flow_uv must have shape [H, W, 2]")

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5

    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_uv_to_colors(u, v, convert_to_bgr)


__all__ = [
    "InputPadder",
    "RAFT",
    "flow_to_image",
]

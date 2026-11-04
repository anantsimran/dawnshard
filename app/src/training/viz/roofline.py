"""Place measured kernels on the device's roofline.

A kernel's time is at least max(FLOPs / peak, bytes / bandwidth). Plotted as achieved
FLOP/s against arithmetic intensity (FLOPs per byte), the two bounds are a slanted
bandwidth roof and a flat compute roof meeting at the ridge point, peak / bandwidth.
A kernel left of the ridge is memory-bound: only moving fewer bytes (fusion, a smaller
dtype) speeds it up.

`main` times five kernels of a decoder training step on CUDA if present, else CPU,
with FLOPs and bytes in closed form, and saves the plot and a CSV:

    uv run python app/src/training/viz/roofline.py --peak-tflops 165 --bandwidth-gbs 1008
"""

import argparse
import csv
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from training.constants import HISTORY_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # noqa: NAR001


@dataclass(frozen=True)
class KernelPoint:
    """One kernel's work, traffic, and measured time."""

    name: str
    flops: float
    bytes_moved: float
    seconds: float

    @property
    def intensity(self) -> float:
        """FLOPs per byte."""
        return self.flops / self.bytes_moved

    @property
    def achieved(self) -> float:
        """FLOP/s achieved."""
        return self.flops / self.seconds


def time_kernel(fn: Callable[[], object], repeats: int = 20) -> float:
    """Return the median seconds of `fn` over `repeats` runs, after 3 warmups, synced."""
    times = []
    for index in range(repeats + 3):  # noqa: NAR001
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        if index >= 3:
            times.append(time.perf_counter() - start)  # noqa: NAR001
    return sorted(times)[len(times) // 2]  # noqa: NAR001


def measure_kernels(
    tokens: int, d: int, ff: int, heads: int, seq: int, dtype: torch.dtype
) -> list[KernelPoint]:
    """Time five kernels of one decoder step at the given sizes.

    Matmul: the MLP up projection. Attention: causal SDPA forward (half the score
    FLOPs are masked). RMSNorm and the SwiGLU elementwise product: read and write
    whole activations. AdamW: one fused step over `d * ff` fp32 parameters.
    """
    size = torch.empty(size=(), dtype=dtype).element_size()
    x = torch.randn(tokens, d, device=DEVICE, dtype=dtype)  # noqa: NAR001
    w = torch.randn(d, ff, device=DEVICE, dtype=dtype)  # noqa: NAR001
    gate, up = (torch.randn(tokens, ff, device=DEVICE, dtype=dtype) for _ in range(2))  # noqa: NAR001
    batch, head_dim = max(tokens // seq, 1), d // heads  # noqa: NAR001
    q = torch.randn(batch, heads, seq, head_dim, device=DEVICE, dtype=dtype)  # noqa: NAR001
    norm = torch.nn.RMSNorm(normalized_shape=d, device=DEVICE, dtype=dtype)
    param = torch.nn.Parameter(data=torch.randn(d * ff, device=DEVICE))  # noqa: NAR001
    param.grad = torch.randn_like(input=param)
    adam = torch.optim.AdamW(params=[param], fused=DEVICE.type == "cuda" or None)
    attention_flops = 4 * batch * heads * seq * seq * head_dim / 2

    def sdpa() -> torch.Tensor:
        """Causal attention forward with q as queries, keys, and values."""
        return F.scaled_dot_product_attention(query=q, key=q, value=q, is_causal=True)

    def swiglu() -> torch.Tensor:
        """The SwiGLU elementwise product."""
        return F.silu(input=gate) * up

    return [
        KernelPoint(
            name="matmul (MLP up)",
            flops=2 * tokens * d * ff,
            bytes_moved=size * (tokens * d + d * ff + tokens * ff),
            seconds=time_kernel(fn=lambda: x @ w),
        ),
        KernelPoint(
            name="attention (SDPA)",
            flops=attention_flops,
            bytes_moved=4 * q.numel() * size,
            seconds=time_kernel(fn=sdpa),
        ),
        KernelPoint(
            name="RMSNorm",
            flops=4 * tokens * d,
            bytes_moved=2 * tokens * d * size,
            seconds=time_kernel(fn=lambda: norm(x)),  # noqa: NAR001
        ),
        KernelPoint(
            name="SwiGLU elementwise",
            flops=5 * tokens * ff,
            bytes_moved=3 * tokens * ff * size,
            seconds=time_kernel(fn=swiglu),
        ),
        KernelPoint(
            name="AdamW step",
            flops=12 * param.numel(),
            bytes_moved=7 * 4 * param.numel(),
            seconds=time_kernel(fn=adam.step),
        ),
    ]


def plot_roofline(
    points: list[KernelPoint], peak_flops: float, bandwidth: float, out_path: Path
) -> None:
    """Save a log-log roofline with `points` on it to `out_path`.

    Side effects:
        Writes the PNG.
    """
    intensities = [p.intensity for p in points] + [peak_flops / bandwidth]
    low, high = min(intensities) / 4, max(intensities) * 4  # noqa: NAR001
    xs = [low * (high / low) ** (i / 99) for i in range(100)]  # noqa: NAR001
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, [min(peak_flops, bandwidth * x) for x in xs], color="black", label="roofline")  # noqa: NAR001
    for point in points:
        ax.scatter(point.intensity, point.achieved, s=40)  # noqa: NAR001
        ax.annotate(  # noqa: NAR001
            point.name,
            (point.intensity, point.achieved),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )  # noqa: NAR001
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="arithmetic intensity (FLOP/byte)",
        ylabel="achieved FLOP/s",
    )
    ax.axvline(x=peak_flops / bandwidth, color="grey", linestyle=":", label="ridge point")
    ax.legend()
    fig.savefig(fname=out_path, bbox_inches="tight", dpi=150)
    plt.close(fig=fig)


def main() -> None:
    """Measure the kernels on this device, then write the roofline PNG and a CSV."""
    parser = argparse.ArgumentParser(description="Place five kernels on the roofline.")
    parser.add_argument("--peak-tflops", type=float, required=True)  # noqa: NAR001
    parser.add_argument("--bandwidth-gbs", type=float, required=True)  # noqa: NAR001
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])  # noqa: NAR001
    parser.add_argument("--tokens", type=int, default=4096)  # noqa: NAR001
    parser.add_argument("--out", type=Path, default=HISTORY_PATH / "roofline.png")  # noqa: NAR001
    args = parser.parse_args()
    points = measure_kernels(
        tokens=args.tokens,
        d=1024,
        ff=3072,
        heads=8,
        seq=min(args.tokens, 2048),  # noqa: NAR001
        dtype=getattr(torch, args.dtype),  # noqa: NAR001
    )  # noqa: NAR001
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_roofline(
        points=points,
        peak_flops=args.peak_tflops * 1e12,
        bandwidth=args.bandwidth_gbs * 1e9,
        out_path=args.out,
    )
    with open(file=args.out.with_suffix(".csv"), mode="w", newline="") as handle:  # noqa: NAR001
        writer = csv.DictWriter(
            f=handle,
            fieldnames=["name", "flops", "bytes_moved", "seconds", "intensity", "achieved"],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(
                rowdict={
                    **asdict(obj=point),
                    "intensity": point.intensity,
                    "achieved": point.achieved,
                }
            )
    print(f"wrote {args.out}")  # noqa: NAR001


if __name__ == "__main__":
    main()

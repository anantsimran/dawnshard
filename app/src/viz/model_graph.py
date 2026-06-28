"""
Visualize PyTorch models — two complementary tools.

─── torchviz: computation graph ────────────────────────────────────────────────
Traces the autograd graph and opens it as an HTML diagram in your browser.
Read the graph bottom-up: blue nodes are parameters (weights/biases), grey boxes
are backward ops (AddmmBackward0 = Linear, ConvolutionBackward0 = Conv2d), orange
boxes are intermediate tensors saved for the backward pass.

    from viz.model_graph import visualize_model
    from model.mnist import CNNClassifier

    visualize_model(model=CNNClassifier(), input_shape=(1, 1, 28, 28))
    # optional: visualize_model(..., output_path=Path("cnn_graph.html"))

─── torchinfo: layer-by-layer table ────────────────────────────────────────────
Prints output shape and parameter count per layer — the fastest way to verify
your architecture matches what you designed.

    from torchinfo import summary
    from model.mnist import MNISTClassifier, CNNClassifier

    summary(model=MNISTClassifier(), input_size=(1, 1, 28, 28))
    summary(model=CNNClassifier(), input_size=(1, 1, 28, 28))

input_size is (batch, channels, height, width). Use batch=1 for a single example.
"""

import base64
import webbrowser
from pathlib import Path

import torch
import torch.nn as nn
from torchviz import make_dot

_DEFAULT_OUTPUT_PATH = Path("/tmp/model_graph.html")  # noqa: NAR001


def visualize_model(
    model: nn.Module,
    input_shape: tuple[int, ...],
    output_path: Path = _DEFAULT_OUTPUT_PATH,
) -> None:
    dummy_input = torch.zeros(input_shape)  # noqa: NAR001
    output = model(dummy_input)  # noqa: NAR001
    dot = make_dot(  # noqa: NAR001
        var=output,
        params=dict(model.named_parameters()),  # noqa: NAR001
        show_attrs=True,
        show_saved=True,
    )
    dot.format = "png"
    png_bytes = dot.pipe()  # noqa: NAR001
    encoded = base64.b64encode(s=png_bytes).decode(encoding="utf-8")
    html = f'<html><body><img src="data:image/png;base64,{encoded}"></body></html>'
    output_path.write_text(data=html)
    webbrowser.open(url=output_path.as_uri())

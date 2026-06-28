"""
mnist_classifier.py
Load, inspect, train, and evaluate on the MNIST dataset using PyTorch.
"""

from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from dataload.constants import DATASETS_CACHE_DIR
from jaxtyping import Float, jaxtyped
from loguru import logger
from setup import DEVICE, init_wandb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train.mean_loss import accumulate_loss, compute_reduced_loss
from train.train_loop import RuntimeConfig, TrainState, fit

IMAGE_SIZE = 28
INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE  # 784 flattened pixels
HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 64
NUM_CLASSES = 10

DEEP_HIDDEN_DIM_1 = 256
DEEP_HIDDEN_DIM_2 = 128


class MNISTClassifier(nn.Module):
    """
    3-layer MLP for 10-class digit classification.

    Architecture:
      Flatten → Linear(784→128) → ReLU → Linear(128→64) → ReLU → Linear(64→10)

    Output: raw logits (no softmax). CrossEntropyLoss applies softmax internally,
    which is numerically more stable than doing it separately.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(  # noqa: NAR001
            nn.Flatten(),
            nn.Linear(in_features=INPUT_DIM, out_features=HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_DIM_1, out_features=HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN_DIM_2, out_features=NUM_CLASSES),
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, f"batch 1 {IMAGE_SIZE} {IMAGE_SIZE}"]
    ) -> Float[torch.Tensor, f"batch {NUM_CLASSES}"]:
        return self.net(x)  # noqa: NAR001


class DeepMNISTClassifier(nn.Module):
    """
    Deep MLP
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=INPUT_DIM, out_features=DEEP_HIDDEN_DIM_1)
        self.fc2 = nn.Linear(in_features=DEEP_HIDDEN_DIM_1, out_features=DEEP_HIDDEN_DIM_2)
        self.fc3 = nn.Linear(in_features=DEEP_HIDDEN_DIM_2, out_features=NUM_CLASSES)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, f"batch 1 {IMAGE_SIZE} {IMAGE_SIZE}"]
    ) -> Float[torch.Tensor, f"batch {NUM_CLASSES}"]:
        x = self.flatten(x)  # noqa: NAR001
        x = F.relu(input=self.fc1(x))
        x = F.relu(input=self.fc2(x))
        return self.fc3(x)  # noqa: NAR001


class ConvolutionalMNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.linear = nn.Linear(in_features=32 * 7 * 7, out_features=NUM_CLASSES)
        self.pool = nn.MaxPool2d(kernel_size=2)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, f"batch 1 {IMAGE_SIZE} {IMAGE_SIZE}"]
    ) -> Float[torch.Tensor, f"batch {NUM_CLASSES}"]:
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        return self.linear(x)  # noqa: NAR001


def get_mnist_dataloader(batch_size: int = 64, data_dir: Path = DATASETS_CACHE_DIR):
    """
    Download MNIST and return (train_loader, test_loader).

    Transform pipeline:
      ToTensor  — uint8 image [0,255] → float32 tensor [0.0,1.0], shape (1,28,28)
      Normalize — (pixel - mean) / std using MNIST's precomputed mean=0.1307, std=0.3081
                  This centers the data around 0, which helps gradient flow
                  during training.
    """
    transform = transforms.Compose(
        transforms=[
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    return (
        DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=2),
    )


def inspect_mnist_dataset(loader: DataLoader):
    """
    Print shape, class info, and one-batch tensor statistics.
    Call this before training to sanity-check your data pipeline.

    Args:
        loader: Any DataLoader wrapping an MNIST dataset.
    """
    ds = cast(typ=datasets.MNIST, val=loader.dataset)
    logger.info("Total samples : {}", len(ds))  # noqa: NAR001
    logger.info("Classes       : {}", ds.classes)  # noqa: NAR001
    logger.info("Image shape   : {}  (C, H, W)", ds[0][0].shape)  # noqa: NAR001

    images, labels = next(iter(loader))  # noqa: NAR001
    logger.info("Batch shape   : {}  (B, C, H, W)", images.shape)  # noqa: NAR001
    logger.info("Label sample  : {}", labels[:8].tolist())  # noqa: NAR001
    logger.info("Pixel range   : [{:.3f}, {:.3f}]", images.min(), images.max())  # noqa: NAR001


def main():
    SHOULD_LOG_WANDB: bool = False
    wandb_run = None
    if SHOULD_LOG_WANDB:
        wandb_run = init_wandb(project="dawnshard", run_name="mnist")

    runtime_config = RuntimeConfig(
        device=DEVICE,
        accumulate_loss=accumulate_loss,
        compute_reduced_loss=compute_reduced_loss,
    )
    model = ConvolutionalMNISTClassifier().to(device=DEVICE)
    train_state = TrainState(
        model=model,
        optimizer=torch.optim.Adam(params=model.parameters(), lr=1e-3),
        criterion=nn.CrossEntropyLoss(),
    )
    train_loader, val_loader = get_mnist_dataloader()
    fit(
        state=train_state,
        config=runtime_config,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=15,
        val_epoch_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        wandb_run=wandb_run,
    )


if __name__ == "__main__":
    main()

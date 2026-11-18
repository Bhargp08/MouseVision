"""ResNet-18 features for the 118 natural-scene templates.

torch is imported inside the functions so that the rest of the package (and CI, which
never installs the ``cnn`` extra) works without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mousevision.config import REPO_ROOT

LAYERS = ("layer1", "layer2", "layer3", "layer4")
FEATURES_PATH = REPO_ROOT / "data" / "features" / "resnet18_ns.npz"
INPUT_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def extract_features(images, grid=(1, 2), batch_size=32):
    """ImageNet ResNet-18 activations, pooled to each grid size and stored as float16.

    ``images`` is uint8 (n, 918, 1174); keys are ``"<layer>_g<grid>"`` plus
    ``image_ids``, ``weights`` and ``torch_version``.
    """
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    images = np.asarray(images)
    if images.ndim != 3:
        raise ValueError("images must have shape (n, height, width)")

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval()

    activations: dict[str, torch.Tensor] = {}

    def capture(name):
        def hook(module, inputs, output):
            activations[name] = output.detach()

        return hook

    handles = [getattr(model, layer).register_forward_hook(capture(layer)) for layer in LAYERS]

    tensor = torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1)
    tensor = tensor.repeat(1, 3, 1, 1)
    tensor = torch.nn.functional.interpolate(
        tensor, size=INPUT_SIZE, mode="bilinear", align_corners=False, antialias=True
    )
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    try:
        with torch.no_grad():
            for start in range(0, tensor.shape[0], batch_size):
                model(tensor[start : start + batch_size])
    finally:
        for handle in handles:
            handle.remove()

    features: dict[str, object] = {}
    for layer in LAYERS:
        activation = activations[layer]
        for size in grid:
            pooled = torch.nn.functional.adaptive_avg_pool2d(activation, (size, size))
            features[f"{layer}_g{size}"] = (
                pooled.flatten(1).numpy().astype(np.float16)
            )

    features["image_ids"] = np.arange(images.shape[0], dtype=np.int64)
    features["weights"] = "IMAGENET1K_V1"
    features["torch_version"] = torch.__version__
    return features


def load_features(path=FEATURES_PATH):
    """Load the cached features; float16 arrays are promoted to float64."""
    with np.load(Path(path), allow_pickle=False) as data:
        out: dict[str, object] = {}
        for key in data.files:
            value = data[key]
            if value.dtype == np.float16:
                out[key] = value.astype(np.float64)
            elif value.shape == ():
                out[key] = value.item()
            else:
                out[key] = value
        return out

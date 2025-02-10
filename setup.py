from setuptools import setup

setup(
    name="aim-thesis",
    install_requires=[
        "wandb",
        "pycocotools",
        "einops",
        "albumentations",
        "accelerate",
    ],
    extras_require={
        "cuda": [
            "torch @ https://download.pytorch.org/whl/cu126/torch-2.x.x+cu126-cp39-cp39-linux_x86_64.whl",
            "torchvision @ https://download.pytorch.org/whl/cu126/torchvision-0.x.x+cu126-cp39-cp39-linux_x86_64.whl",
            "torchaudio @ https://download.pytorch.org/whl/cu126/torchaudio-2.x.x+cu126-cp39-cp39-linux_x86_64.whl",
        ],
        "cpu": [
            "torch",
            "torchvision",
            "torchaudio",
        ],
    }
)
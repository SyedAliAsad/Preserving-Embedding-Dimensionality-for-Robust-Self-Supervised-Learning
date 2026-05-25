from setuptools import setup, find_packages

setup(
    name="dim-ssl",
    version="0.1.0",
    description="Dimensionality-Preserving Self-Supervised Learning",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13",
        "torchvision>=0.14",
        "pyyaml>=6.0",
        "wandb>=0.15",
        "numpy>=1.21",
        "scipy>=1.7",
        "tqdm",
        "matplotlib",
        "scikit-learn",
    ],
)

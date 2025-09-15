import torch

from src.utils import select_backend


def main():
    print(f"torch version: {torch.__version__}")
    device = select_backend()
    if device.type == "cuda":
        print("CUDA is available")
    elif device.type == "mps":
        print("Metal is available")
    else:
        print("CPU is available")


if __name__ == "__main__":
    main()

import torch


def select_backend() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


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

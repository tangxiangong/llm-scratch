import os

import torch


def select_backend() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_data() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "..", "..", "data", "the-verdict.txt")

    with open(data_file, "r", encoding="utf-8") as f:
        raw_text = f.read()
    return raw_text


if __name__ == "__main__":
    raw_text = load_data()
    print("Total number of character:", len(raw_text))
    print(raw_text[:100])

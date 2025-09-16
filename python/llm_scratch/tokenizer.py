import tiktoken

from .utils import load_data


class GPTTokenizer(object):
    def __init__(self):
        self.tokenizer: tiktoken.Encoding = tiktoken.encoding_for_model("gpt-2")
        self.special_tokens: set[str] | None = None

    def add_special_tokens(self, special_tokens: list[str]):
        self.special_tokens = set(special_tokens)

    def encode(self, text: str):
        if self.special_tokens is None:
            return self.tokenizer.encode(text, disallowed_special="all")
        else:
            return self.tokenizer.encode(text, allowed_special=self.special_tokens)


def tokenization() -> list[int]:
    raw_text = load_data()
    tokenizer = GPTTokenizer()
    enc_text = tokenizer.encode(raw_text)
    return enc_text


if __name__ == "__main__":
    ids = tokenization()
    print(ids)

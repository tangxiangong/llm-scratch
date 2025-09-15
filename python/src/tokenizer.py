import tiktoken


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

if __name__ == "__main__":

    def test_tokenizer():
        tokenizer = GPTTokenizer()
        tokenizer.add_special_tokens(["<|endoftext|>"])
        text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunkownPlace."
        integers = tokenizer.encode(text)
        print(integers)

    test_tokenizer()

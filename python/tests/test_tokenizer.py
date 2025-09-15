from src.tokenizer import GPTTokenizer


def test_gpt_tokenizer():
    tokenizer = GPTTokenizer()
    tokenizer.add_special_tokens(["<|endoftext|>"])
    text = "Hello, do you like tea?  In the sunlit terraces of someunkownPlace."
    integers = tokenizer.encode(text)
    print(integers)

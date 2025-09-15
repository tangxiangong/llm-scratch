if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "..", "..", "data", "the-verdict.txt")

    with open(data_file, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of character:", len(raw_text))
    print(raw_text[:100])

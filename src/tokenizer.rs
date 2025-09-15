use anyhow::Result;
use std::ops::Deref;
use tokenizers::{AddedToken, Tokenizer};

/// GPT-2 tokenizer
pub struct GPTTokenizer {
    tokenizer: Tokenizer,
    special_tokens: Vec<AddedToken>,
}

impl Deref for GPTTokenizer {
    type Target = Tokenizer;

    fn deref(&self) -> &Self::Target {
        &self.tokenizer
    }
}

impl GPTTokenizer {
    pub fn new() -> Result<Self> {
        let tokenizer =
            Tokenizer::from_pretrained("gpt2", None).map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(Self {
            tokenizer,
            special_tokens: vec![],
        })
    }

    /// Add special tokens to the tokenizer.
    pub fn add_special_tokens(mut self, special_tokens: &[&str]) -> Self {
        let tokens = special_tokens.iter().map(|&s| AddedToken::from(s, true));
        self.special_tokens.extend(tokens);
        self.tokenizer.add_special_tokens(&self.special_tokens);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer() {
        let tokenizer = GPTTokenizer::new()
            .unwrap()
            .add_special_tokens(&["<|endoftext|>"]);
        let tokens = tokenizer
            .encode(
                "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunkownPlace.",
                true,
            )
            .unwrap();
        println!("{:?}", tokens.get_ids());
    }
}

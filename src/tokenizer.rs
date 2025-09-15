#[cfg(test)]
mod tests {
    #[test]
    fn test_reader() {
        let raw_text = std::fs::read_to_string("./data/the-verdict.txt").unwrap();
        println!("Total number of character: {}", raw_text.len());
        println!("{:?}", &raw_text[..100]);
    }
}

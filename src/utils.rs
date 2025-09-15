use anyhow::Context;
use candle_core::Device;

pub fn select_backend() -> Device {
    if let Ok(device) = Device::new_cuda(0) {
        device
    } else if let Ok(device) = Device::new_metal(0) {
        device
    } else {
        Device::Cpu
    }
}

pub fn load_data() -> anyhow::Result<String> {
    let raw_text = std::fs::read_to_string("./data/the-verdict.txt")
        .with_context(|| r#"Could not read data file "./data/the-verdict.txt""#)?;
    Ok(raw_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_data() {
        let raw_text = load_data().unwrap();
        println!("Total number of character: {}", raw_text.len());
        println!("{:?}", &raw_text[..100]);
    }
}

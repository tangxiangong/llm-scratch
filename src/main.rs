use candle_core::{Device, Result, Tensor};

fn main() -> Result<()> {
    let device = Device::new_metal(0)?;
    let tensor1 = Tensor::rand(0.0f32, 1.0, (2, 3), &device)?;
    let tensor2 = Tensor::rand(0.0f32, 1.0, (2, 3), &device)?;
    let tensor = (tensor1 + tensor2)?;
    println!("tensor: {:?}\n {:?}", tensor, tensor.to_vec2::<f32>()?);
    Ok(())
}

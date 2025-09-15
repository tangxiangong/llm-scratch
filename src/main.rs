use candle_core::Device;
use llm_scratch::utils::select_backend;

fn main() {
    println!("candle verison: 0.9.1");
    let device = select_backend();
    match device {
        Device::Cuda(_) => println!("CUDA is available"),
        Device::Metal(_) => println!("Metal is available"),
        Device::Cpu => println!("CPU is available"),
    }
}

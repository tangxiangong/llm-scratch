use candle_core::Device;

fn select_backend() -> Device {
    if let Ok(device) = Device::new_cuda(0) {
        device
    } else if let Ok(device) = Device::new_metal(0) {
        device
    } else {
        Device::Cpu
    }
}

fn main() {
    println!("candle verison: 0.9.1");
    let device = select_backend();
    match device {
        Device::Cuda(_) => println!("CUDA is available"),
        Device::Metal(_) => println!("Metal is available"),
        Device::Cpu => println!("CPU is available"),
    }
}

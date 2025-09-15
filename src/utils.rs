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

use std::{sync::{Arc, Mutex}, thread::sleep, time::Duration};

use buffer_display as bd;
use bd::BufferDisplay;

// Example copied from the crate test function.
fn main() {
    let pixels: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new((0..1440*900*4).map(|_| 0u8).collect()));
    let ptr_clone = pixels.clone();

    println!("");
    println!("The window should go from black to white");
    
    let display_thread = std::thread::spawn(move || {
        println!("Visually displaying the buffer in a window, while it is being updated.");
        println!("");
        let mut d = BufferDisplay::init(1440, 900, ptr_clone);
        d.run();
    });
    
    println!("Updating the buffer independently. Here: on the main thread.");
    let mut temp: Vec<u8>;
    for i in 1..256 {
        sleep(Duration::from_millis(10)); // Swapchain ???
        temp = (0..1440*900*4).map(|_| i as u8).collect::<Vec<u8>>();
        {
            let mut write = pixels.lock().unwrap();
            let buf = write.as_mut();
            *buf = temp;
        }
    }
    
    display_thread.join().unwrap();

    println!("");
    println!("Finished. Should be caused by you closing the window.");
}
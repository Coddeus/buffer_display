use std::sync::{Arc, Mutex};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, BlitImageInfo, BufferImageCopy, ClearColorImageInfo, CommandBufferUsage, CopyBufferToImageInfo, PrimaryCommandBufferAbstract
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType}, Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags
    },
    format::Format,
    image::{
        sampler::Filter,
        view::ImageView,
        Image, ImageCreateInfo, ImageLayout, ImageUsage,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    DeviceSize, Validated, VulkanError, VulkanLibrary,
};
use winit::{
    dpi::LogicalSize, event::{Event, WindowEvent}, event_loop::{ControlFlow, EventLoop, EventLoopBuilder}, platform::run_return::EventLoopExtRunReturn, window::{Window, WindowBuilder}
};
use winit::platform::windows::EventLoopBuilderExtWindows;

/// Displays a 2D RGBA u8 buffer from its pointer and size.
pub struct BufferDisplay {
    event_loop: EventLoop<()>,
    vk: Vk,
}

/// Vulkan backend implementation stuff.
struct Vk {
    _instance: Arc<Instance>,
    _surface: Arc<Surface>,
    _physical_device: Arc<PhysicalDevice>,
    _queue_family_index: u32,

    window: Arc<Window>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    command_buffer_allocator: StandardCommandBufferAllocator,

    viewport: Viewport,
    framebuffers: Vec<Arc<Framebuffer>>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    
    intermediary_image: Arc<Image>,
    upload_buffer: Subbuffer<[u8]>,

    swapchain: Arc<Swapchain>,
    images: Vec<Arc<Image>>,

    width: u32,
    height: u32,
    data_pointer: Arc<Mutex<Vec<u8>>>,
}

const WIDTH: u32 = 1440;
const HEIGHT: u32 = 900;
impl BufferDisplay {
    pub fn init(width: u32, height: u32, data_pointer: Arc<Mutex<Vec<u8>>>) -> BufferDisplay {
        let event_loop = EventLoopBuilder::new().with_any_thread(true).build();
    
        let library = VulkanLibrary::new().unwrap();
        let required_extensions = Surface::required_extensions(&event_loop);
        let _instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let window = Arc::new(WindowBuilder::new().with_inner_size(LogicalSize::new(WIDTH, HEIGHT)).with_title("Buffer display").build(&event_loop).unwrap());
        let _surface = Surface::from_window(_instance.clone(), window.clone()).unwrap();
    
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let (_physical_device, _queue_family_index) = _instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &_surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();
    
        println!(
            "Using device: {} (type: {:?})",
            _physical_device.properties().device_name,
            _physical_device.properties().device_type,
        );
    
        let (device, mut queues) = Device::new(
            _physical_device.clone(),
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: _queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();
        let queue = queues.next().unwrap();
    
        let (swapchain, images) = {
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&_surface, Default::default())
                .unwrap();
            let image_format = device
                .physical_device()
                .surface_formats(&_surface, Default::default())
                .unwrap()[0]
                .0;

            Swapchain::new(
                device.clone(),
                _surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    present_mode: vulkano::swapchain::PresentMode::Immediate,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap()
        };
    
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());
    
        let render_pass = vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();
    
        let upload_buffer: Subbuffer<[u8]> = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            (width * height * 4) as DeviceSize,
        )
        .unwrap();
    
        let uploads = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
    
        
        let intermediary_image: Arc<Image> = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                format: Format::R8G8B8A8_UNORM,
                extent: [width, height, 1],
                usage: ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();

        let mut viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [0.0, 0.0],
            depth_range: 0.0..=1.0,
        };
        let framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);
    
        let recreate_swapchain = false;
        let previous_frame_end = Some(
            uploads
                .build()
                .unwrap()
                .execute(queue.clone())
                .unwrap()
                .boxed(),
        );
    
        Self {
            event_loop,
            vk: Vk {
                _instance,
                _surface,
                _physical_device,
                _queue_family_index,

                window,
                device,
                queue,
                render_pass,
                command_buffer_allocator,

                viewport,
                framebuffers,
                recreate_swapchain,
                previous_frame_end,

                intermediary_image,
                upload_buffer,

                swapchain,
                images,
                
                width,
                height,
                data_pointer,
            }
        }
    }

    pub fn run(&mut self) {
        self.vk.run(&mut self.event_loop);
    }
}

impl Vk {
    fn run(&mut self, event_loop: &mut EventLoop<()>) {
        
    event_loop.run_return(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            self.recreate_swapchain = true;
        }
        Event::RedrawEventsCleared => {
            let image_extent: [u32; 2] = self.window.inner_size().into();

            if image_extent.contains(&0) {
                return;
            }

            self.previous_frame_end.as_mut().unwrap().cleanup_finished();
            
            if self.recreate_swapchain {
                (self.swapchain, self.images) = self.swapchain
                    .recreate(SwapchainCreateInfo {
                        image_extent,
                        ..self.swapchain.create_info()
                    })
                    .expect("failed to recreate swapchain");

                self.framebuffers =
                    window_size_dependent_setup(&self.images, self.render_pass.clone(), &mut self.viewport);
                self.recreate_swapchain = false;
            }

            let (image_index, suboptimal, acquire_future) =
                match acquire_next_image(self.swapchain.clone(), None).map_err(Validated::unwrap) {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        self.recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

            if suboptimal {
                self.recreate_swapchain = true;
            }

            let mut uploads = AutoCommandBufferBuilder::primary(
                &self.command_buffer_allocator,
                self.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            let buf_size = [self.width, self.height];
        
            uploads
                .clear_color_image(ClearColorImageInfo::image(self.intermediary_image.clone()))
                .unwrap()
                .copy_buffer_to_image(CopyBufferToImageInfo {
                    regions: [BufferImageCopy {
                        image_subresource: self.intermediary_image.subresource_layers(),
                        image_extent: [buf_size[0], buf_size[1], 1],
                        ..Default::default()
                    }]
                    .into(),
                    ..CopyBufferToImageInfo::buffer_image(self.upload_buffer.clone(), self.intermediary_image.clone())
                })
                .unwrap()
                .blit_image(BlitImageInfo {
                    src_image_layout: ImageLayout::TransferSrcOptimal,
                    dst_image_layout: ImageLayout::TransferDstOptimal,
                    filter: Filter::Linear,
                    ..BlitImageInfo::images(self.intermediary_image.clone(), self.images[image_index as usize].clone())
                })
                .unwrap();
            let up = uploads.build().unwrap();
            {
                let mut write_guard = self.upload_buffer.write().unwrap();
        
                for (o, i) in write_guard.iter_mut().zip(self.data_pointer.lock().unwrap().iter()) {
                    *o = *i;
                }
            }

            let future = self.previous_frame_end
                .take()
                .unwrap()
                .join(acquire_future)
                .then_execute(self.queue.clone(), up)
                .unwrap()
                .then_swapchain_present(
                    self.queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
                )
                .then_signal_fence_and_flush();

            match future.map_err(Validated::unwrap) {
                Ok(future) => {
                    future.wait(None).unwrap();
                    self.previous_frame_end = Some(future.boxed());
                }
                Err(VulkanError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                }
                Err(e) => {
                    println!("failed to flush future: {e}");
                    self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                }
            }
            
        }
        _ => (),
    });
    }
}

fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let extent = images[0].extent();
    viewport.extent = [extent[0] as f32, extent[1] as f32];

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::{thread::sleep, time::Duration};

    #[test]
    #[allow(non_snake_case)]
    fn VISUALTEST_realtime_display_buffer() {
        let pixels: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new((0..1440*900*4).map(|_| 0u8).collect()));
        let ptr_clone = pixels.clone();
        
        let display_thread = std::thread::spawn(move || {
            let mut d = BufferDisplay::init(1440, 900, ptr_clone);
            d.run();
        });
        
        let mut temp: Vec<u8>;
        for i in 1..256 {
            sleep(Duration::from_millis(10));
            temp = (0..1440*900*4).map(|_| i as u8).collect::<Vec<u8>>();
            {
                let mut write = pixels.lock().unwrap();
                let buf = write.as_mut();
                *buf = temp;
            }
        }
        
        display_thread.join().unwrap();
    }
}
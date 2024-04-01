# Buffer Display
Displays a buffer on a window in realtime.

Displays a 2D RGBA u8 buffer from its width/height as `u32`s and a pointer to it as an `Arc<Mutex<Vec<u8>>>`.  
Works when launched from another thread, and updates in realtime the buffer behind the pointer given.

Made for Windows, may be extended in the future.

Originally made to watch fractals rendering, see my [Fractals](https://github.com/Coddeus/Fractals) repo.

## License
GNU GPL v3, see [LICENSE](./LICENSE).
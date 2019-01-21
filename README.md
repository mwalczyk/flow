# flow
🔮 A GPU-based progressive path tracer written in Vulkan. 


## Description
For the time being, `flow` uses a single render pass instance with two distinct subpasses. The first subpass involves two floating-point images in a "ping-pong" arrangement. This is how the light accumulates over the course of many frames, leading to a well-converged image. Whichever of the two images was used as a color attachment during this first subpass serves as an input attachment to the next (and final) subpass. Input attachments are unique to Vulkan and allow a render pass attachment to be read in a fragment shader stage during a subpass. Input attachments come with several restrictions and do not support random access like a typical `sampler2D`, for example.

The second subpass simply reads from this input attachment and writes to one of the swapchain images that are presented to the screen. 

As such, there are two separate graphics pipelines - one that runs the main path tracing routine and another that normalizes the accumulated light (converts it from HDR to `[0..1]`) and writes to the corresponding swapchain image.

Built on top of [vkstarter](https://github.com/mwalczyk/vkstarter).

## Tested On
- Windows 8.1, Windows 10, Windows 7
- NVIDIA GeForce GTX 1050ti
- Vulkan SDK `1.1.70.1`
- Visual Studio 2015.

## To Build
1. Clone this repo.
2. Inside the repo, create a new folder named `third_party`.
3. Download the [Vulkan SDK for Windows](https://vulkan.lunarg.com/sdk/home#windows). Make sure the `VK_SDK_PATH` environment
   variable is defined on your system.
4. Download the [GLFW pre-compiled binaries](http://www.glfw.org/download.html) (64-bit Windows) and place inside the `third_party` directory. Rename this folder to `glfw`.
5. Optionally, run `vkstarter/compile.bat` to convert the included `GLSL` shaders to `SPIR-V`. This will be run automatically as a pre-build event by Visual Studio.
6. Open the Visual Studio 2015 solution file.
7. Build the included project.

NOTE: There appears to be a bug in `vulkan.hpp`, which requires one to change line `35431` from:
```cpp
ObjectDestroy<NoParent> deleter( *this, allocator );
```
to:
```cpp
ObjectDestroy<NoParent> deleter( allocator );
```

## To Do
- [ ] Tone mapping and exposure adjustment
- [ ] Explicit light sampling 
- [ ] Russian roulette path termination
- [x] Reflective materials (dielectrics)
- [ ] Improved BRDFs (GGX, Cook-Torrance, etc.)
- [ ] Spatial acceleration data structures (most likely some form of GPU KD-tree)
- [ ] Scene format (`.json`) and parser
- [ ] Screenshot utility

### License

:copyright: The Interaction Department, tigrazone 2018

[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)


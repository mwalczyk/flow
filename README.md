# flow
🔮 A GPU-based progressive path tracer written in Vulkan. 

<p align="center">
  <img src="https://github.com/mwalczyk/flow/blob/master/screenshots/screenshot.png" alt="screenshot" width="300" height="auto"/>
</p>

## Description
For the time being, `flow` uses a single render pass instance with two distinct subpasses. The first subpass involves two floating-point images in a "ping-pong" arrangement. This is how the light accumulates over the course of many frames, leading to a well-converged image. Whichever of the two images was used as a color attachment during this first subpass serves as an input attachment to the next (and final) subpass. Input attachments are unique to Vulkan and allow a render pass attachment to be read in a fragment shader stage during a subpass. Input attachments come with several restrictions and do not support random access like a typical `sampler2D`, for example.

The second subpass simply reads from this input attachment and writes to one of the swapchain images that are presented to the screen. 

As such, there are two separate graphics pipelines - one that runs the main path tracing routine and another that normalizes the accumulated light (converts it from HDR to `[0..1]`) and writes to the corresponding swapchain image.

Built on top of [vkstarter](https://github.com/mwalczyk/vkstarter).

## Tested On
- Ubuntu 18.04
- NVIDIA GeForce GTX 1070
- Vulkan SDK `1.1.106.0`

## To Build
1. Clone this repo.
2. Clone GLFW (used for windowing): 
```
git clone https://github.com/glfw/glfw.git third_party/glfw
```
3. Download the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) for your OS. Make sure the `VULKAN_SDK` environment variable is defined on your system.
4. Compile the included shader files using `glslangValidator`:
```
cd flow
sh ./compile_shaders.sh
```
4. Finally, from the root directory, run the following commands:
```
mkdir build
cd build
cmake ..
make

./Flow
```

## To Do
- [ ] Tone mapping and exposure adjustment
- [x] Explicit light sampling 
- [x] Russian roulette path termination
- [ ] Refractive materials (dielectrics)
- [ ] Improved BRDFs (GGX, Cook-Torrance, etc.)
- [ ] Spatial acceleration data structures (most likely some form of GPU BVH)
- [ ] Scene format (`.json`) and parser
- [ ] Screenshot utility

### License

[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)


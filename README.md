# flow
🔮 A GPU-based progressive path tracer written in Vulkan. 

<p>
  <img src="https://github.com/mwalczyk/flow/blob/master/screenshots/screenshot.png" alt="screenshot" width="300" height="auto"/>
</p>

## Description
Built on top of [vkstarter](https://github.com/mwalczyk/vkstarter).

## Tested On
- Windows 8.1, Windows 10
- NVIDIA GeForce GTX 970M, NVIDIA GeForce GTX 980
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

### License

:copyright: The Interaction Department 2018

[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)


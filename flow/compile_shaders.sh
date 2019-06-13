#!/bin/bash

$VULKAN_SDK/bin/glslangValidator -V -o shaders/quad.spv shaders/quad.vert
$VULKAN_SDK/bin/glslangValidator -V -o shaders/pathtrace.spv shaders/pathtrace.frag
$VULKAN_SDK/bin/glslangValidator -V -o shaders/composite.spv shaders/composite.frag
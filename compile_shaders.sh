#!/bin/bash

$VULKAN_SDK/bin/glslangValidator -V -o flow/shaders/quad.spv flow/shaders/quad.vert
$VULKAN_SDK/bin/glslangValidator -V -o flow/shaders/pathtrace.spv flow/shaders/pathtrace.frag
$VULKAN_SDK/bin/glslangValidator -V -o flow/shaders/composite.spv flow/shaders/composite.frag
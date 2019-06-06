#!/bin/bash

$VULKAN_SDK/bin/glslangValidator -V -o quad.spv quad.vert
$VULKAN_SDK/bin/glslangValidator -V -o pathtrace.spv pathtrace.frag
$VULKAN_SDK/bin/glslangValidator -V -o composite.spv composite.frag
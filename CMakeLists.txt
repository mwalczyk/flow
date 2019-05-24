cmake_minimum_required (VERSION 2.6)
project (Flow)

# enable C++11
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-std=c++11)

# include flow source files
include_directories("${PROJECT_SOURCE_DIR}/flow")
include_directories("${PROJECT_SOURCE_DIR}/third_party/glfw/include/GLFW")

file(GLOB SOURCES "flow/*.cpp")
add_executable(Flow ${SOURCES})

# as of version 3.7, CMake supports Vulkan out-of-the-box
find_package(Vulkan REQUIRED)
target_link_libraries(Flow Vulkan::Vulkan)



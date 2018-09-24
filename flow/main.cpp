#include <iostream>
#include <sstream>
#include <fstream>
#include <limits>
#include <chrono>
#include <functional>

#define NOMINMAX
#define GLFW_EXPOSE_NATIVE_WIN32
#include "glfw3.h"
#include "glfw3native.h"

#define VK_USE_PLATFORM_WIN32_KHR
#include "vulkan/vulkan.hpp"

//#define _DEBUG

#ifdef _DEBUG
#define LOG_DEBUG(x) std::cout << x << "\n"
#else
#define LOG_DEBUG(x) 
#endif



#include <time.h>

clock_t startTime = clock(), timeNow;
char strrr[1000];

char *fn_prefix = "img";

float prvTime = -1;
float lastSaveTime = 0;
float timeSecs;

uint32_t ttl_passes;


int maxTime=0, maxPasses=0;
int saveEveryTime=0, saveEveryPasses=0;



struct alignas(8) PushConstants
{
	float time;
	float frame_counter;
	float resolution[2];
	float mouse[2];
	float mouse_down;
	/* Add more members here: mind the struct alignment */
};

float get_elapsed_time()
{
	static std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

	return static_cast<float>(ms) / 1000.0f;
}



void instantSave()
{
	snprintf(strrr, 1000,
				"%s%ld-%.2fs.png",
				fn_prefix,
				ttl_passes,
				timeSecs
			);
			
	printf("SAVE %s... ", strrr);
	printf("\n");
}


vk::UniqueShaderModule load_spv_into_module(const vk::UniqueDevice& device, const std::string& filename) 
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) 
	{
		throw std::runtime_error("Failed to open file");
	}

	size_t file_size = static_cast<size_t>(file.tellg());
	std::vector<char> buffer(file_size);
	file.seekg(0);
	file.read(buffer.data(), file_size);
	file.close();

	auto shader_module_create_info = vk::ShaderModuleCreateInfo{ {}, static_cast<uint32_t>(buffer.size()), reinterpret_cast<const uint32_t*>(buffer.data()) };
	
	return device->createShaderModuleUnique(shader_module_create_info);
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT object_type, uint64_t object, size_t location, int32_t code, const char* layer_prefix, const char* msg, void* user_data) 
{
	std::ostringstream message;

	if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT) 
	{
		message << "ERROR: ";
	} 
	else if (flags & VK_DEBUG_REPORT_WARNING_BIT_EXT) 
	{
		message << "WARNING: ";
	} 
	else if (flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT) 
	{
		message << "PERFORMANCE WARNING: ";
	} 
	else if (flags & VK_DEBUG_REPORT_INFORMATION_BIT_EXT) 
	{
		message << "INFO: ";
	} 
	else if (flags & VK_DEBUG_REPORT_DEBUG_BIT_EXT) 
	{
		message << "DEBUG: ";
	}
	message << "[" << layer_prefix << "] Code " << code << " : " << msg;
	std::cout << message.str() << std::endl;

	return VK_FALSE;
}

class Application
{
public:
	Application(uint32_t width, uint32_t height, const std::string& name) :
		width{ width }, 
		height{ height }, 
		name{ name }, 
		samples_per_pixel{ 0 },
		total_frames_elapsed{ 0 },
		cursor_position{ 0.5f, 0.5f },
		swapchain_image_format{ vk::Format::eB8G8R8A8Unorm },
		ping_pong_image_format{ vk::Format::eR32G32B32A32Sfloat }
	{
		setup();
		ttl_passes = 0;
	}

	~Application()
	{	
		// Wait for all work on the GPU to finish
		device->waitIdle();

		// Clean up GLFW objects
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	static void on_window_resized(GLFWwindow* window, int width, int height) 
	{
		Application* app = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
		app->resize();
	}

	static void on_cursor_moved(GLFWwindow* window, double xpos, double ypos)
	{
		if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
		{
			Application* app = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
			app->cursor_position[0] = static_cast<float>(xpos) / static_cast<float>(app->width);
			app->cursor_position[1] = static_cast<float>(ypos) / static_cast<float>(app->height);
		}
	}

	void resize()
	{
		device->waitIdle();

		int new_width;
		int new_height;
		glfwGetWindowSize(window, &new_width, &new_height);
		width = new_width;
		height = new_height;
		LOG_DEBUG("Window resized to " + std::to_string(width) + " x " + std::to_string(height));

		swapchain.reset();
		render_pass.reset();
		pipeline_pathtrace.reset();
		pipeline_composite.reset();

		device_memory_ab.reset();
		descriptor_set_a.reset();
		descriptor_set_b.reset();
		image_a.reset();
		image_b.reset();
		image_view_a.reset();
		image_view_b.reset();

		// We do not need to explicitly clear the framebuffers or swapchain image views, since that is taken
		// care of by the `initialize_*()` methods below

		initialize_swapchain();
		initialize_render_pass();
		initialize_pipelines();
		initialize_ping_pong_images();
		initialize_descriptor_sets();
		initialize_framebuffers();

		clear_ping_pong_images();
	}

	void setup()
	{
		// General setup 
		initialize_window();
		initialize_instance();
		initialize_device();
		initialize_surface();
		initialize_swapchain();
		initialize_render_pass();
		initialize_descriptor_set_layout();
		initialize_pipeline_layout();
		initialize_pipelines();
		
		// We need to do this before initializing our framebuffers, since each framebuffer will be associated with either A or B
		initialize_ping_pong_images();

		// Per-frame resources
		initialize_framebuffers();
		initialize_command_pool();
		initialize_command_buffers();
		initialize_synchronization_primitives();
		clear_ping_pong_images();

		// Descriptors
		initialize_descriptor_pool();
		initialize_sampler();
		initialize_descriptor_sets();
	}

	void initialize_window()
	{
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(width, height, name.c_str(), nullptr, nullptr);

		glfwSetWindowUserPointer(window, this);
		glfwSetWindowSizeCallback(window, on_window_resized);
		glfwSetCursorPosCallback(window, on_cursor_moved);
	}

	void initialize_instance()
	{
		std::vector<const char*> layers;
		std::vector<const char*> extensions{ VK_EXT_DEBUG_REPORT_EXTENSION_NAME, VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME };
#ifdef _DEBUG
		layers.push_back("VK_LAYER_LUNARG_standard_validation");
#endif
		
		auto application_info = vk::ApplicationInfo{ name.c_str(), VK_MAKE_VERSION(1, 0, 0), name.c_str(), VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_1 };

		instance = vk::createInstanceUnique(vk::InstanceCreateInfo{ {}, &application_info, static_cast<uint32_t>(layers.size()), layers.data(), static_cast<uint32_t>(extensions.size()), extensions.data() });

#ifdef _DEBUG
		auto dynamic_dispatch_loader = vk::DispatchLoaderDynamic{ instance.get() };
		auto debug_report_callback_create_info = vk::DebugReportCallbackCreateInfoEXT{ vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning, debug_callback };

		debug_report_callback = instance->createDebugReportCallbackEXT(debug_report_callback_create_info, nullptr, dynamic_dispatch_loader);
		LOG_DEBUG("Initializing debug report callback");
#endif
	}

	void initialize_device()
	{
		// First, we select a physical device
		auto physical_devices = instance->enumeratePhysicalDevices();
		assert(!physical_devices.empty());
		physical_device = physical_devices[0];

		auto queue_family_properties = physical_device.getQueueFamilyProperties();

		const float priority = 0.0f;
		auto predicate = [](const vk::QueueFamilyProperties& item) { return item.queueFlags & vk::QueueFlagBits::eGraphics; };
		auto queue_create_info = vk::DeviceQueueCreateInfo{}
			.setPQueuePriorities(&priority)
			.setQueueCount(1)
			.setQueueFamilyIndex(static_cast<uint32_t>(std::distance(queue_family_properties.begin(), std::find_if(queue_family_properties.begin(), queue_family_properties.end(), predicate))));
		LOG_DEBUG("Using queue family at index [ " << queue_create_info.queueFamilyIndex << " ], which supports graphics operations");

		// Save the index of the chosen queue family
		queue_family_index = queue_create_info.queueFamilyIndex;

		// Then, we construct a logical device around the chosen physical device
		const std::vector<const char*> device_extensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
		auto device_create_info = vk::DeviceCreateInfo{}
			.setPQueueCreateInfos(&queue_create_info)
			.setQueueCreateInfoCount(1)
			.setPpEnabledExtensionNames(device_extensions.data())
			.setEnabledExtensionCount(static_cast<uint32_t>(device_extensions.size()));

		device = physical_device.createDeviceUnique(device_create_info);

		const uint32_t queue_index = 0;
		queue = device->getQueue(queue_family_index, queue_index);
	}

	void initialize_surface()
	{	
		auto surface_create_info = vk::Win32SurfaceCreateInfoKHR{ {}, GetModuleHandle(nullptr), glfwGetWin32Window(window) };

		surface = instance->createWin32SurfaceKHRUnique(surface_create_info);
	}

	void initialize_swapchain()
	{
		surface_capabilities = physical_device.getSurfaceCapabilitiesKHR(surface.get());
		surface_formats = physical_device.getSurfaceFormatsKHR(surface.get());
		surface_present_modes = physical_device.getSurfacePresentModesKHR(surface.get());
		auto surface_support = physical_device.getSurfaceSupportKHR(queue_family_index, surface.get());

		swapchain_extent = vk::Extent2D{ width, height };

		auto swapchain_create_info = vk::SwapchainCreateInfoKHR{}
			.setPresentMode(vk::PresentModeKHR::eMailbox)
			.setImageExtent(swapchain_extent)
			.setImageFormat(swapchain_image_format)
			.setImageArrayLayers(1)
			.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
			.setMinImageCount(surface_capabilities.minImageCount + 1)
			.setPreTransform(surface_capabilities.currentTransform)
			.setClipped(true)
			.setSurface(surface.get());

		swapchain = device->createSwapchainKHRUnique(swapchain_create_info);

		// Retrieve the images from the swapchain
		swapchain_images = device->getSwapchainImagesKHR(swapchain.get());
		LOG_DEBUG("There are [ " << swapchain_images.size() << " ] images in the swapchain");

		// Create an image view for each image in the swapchain
		swapchain_image_views.clear();

		const auto subresource_range = vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
		for (const auto& image : swapchain_images)
		{
			auto image_view_create_info = vk::ImageViewCreateInfo{ {}, image, vk::ImageViewType::e2D, swapchain_image_format, {}, subresource_range };
			swapchain_image_views.push_back(device->createImageViewUnique(image_view_create_info));
		}
		LOG_DEBUG("Created [ " << swapchain_image_views.size() << " ] image views");
	}

	void initialize_render_pass()
	{
		// The image that will be rendered into (either A or B)
		auto attachment_description_pong = vk::AttachmentDescription{}
			.setFormat(ping_pong_image_format)
			.setLoadOp(vk::AttachmentLoadOp::eLoad) // Don't erase what was previously drawn into this attachment
			.setStoreOp(vk::AttachmentStoreOp::eStore)
			.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
			.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
			.setInitialLayout(vk::ImageLayout::eGeneral) 
			.setFinalLayout(vk::ImageLayout::eGeneral); 
		// TODO: the layouts above don't seem correct: when the renderpass begins, the image should be a color attachment and when it
		// ends, it should be shader read-only

		// The final image that will be show on the screen
		auto attachment_description_swap = vk::AttachmentDescription{}
			.setFormat(swapchain_image_format)
			.setLoadOp(vk::AttachmentLoadOp::eClear)
			.setStoreOp(vk::AttachmentStoreOp::eStore)
			.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
			.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
			.setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

		vk::AttachmentDescription all_attachment_descriptions[2] = 
		{ 
			attachment_description_pong, // Used by the first and second subpasses	
			attachment_description_swap	 // Only used by the second subpass (for presentation)
		};

		// Indices in the array of `vk::AttachmentDescription` above
		const uint32_t pong_index = 0;
		const uint32_t swap_index = 1;

		// Subpass #1
		vk::AttachmentReference attachment_references_pong[1] = 
		{
			vk::AttachmentReference{ pong_index, vk::ImageLayout::eColorAttachmentOptimal } // Color attachment
		};
		auto subpass_description_pong = vk::SubpassDescription{}
			.setPColorAttachments(attachment_references_pong)
			.setColorAttachmentCount(1);

		// Subpass #2
		vk::AttachmentReference attachment_references_final[2] = 
		{
			vk::AttachmentReference{ swap_index, vk::ImageLayout::eColorAttachmentOptimal }, // Color attachment
			vk::AttachmentReference{ pong_index, vk::ImageLayout::eShaderReadOnlyOptimal } // Input attachment
		};
		auto subpass_description_final = vk::SubpassDescription{}
			.setPColorAttachments(&attachment_references_final[0])
			.setColorAttachmentCount(1)
			.setPInputAttachments(&attachment_references_final[1])
			.setInputAttachmentCount(1);

		// TODO: why do we have this...?
		auto subpass_dependency_ext_0 = vk::SubpassDependency{}
			.setSrcSubpass(VK_SUBPASS_EXTERNAL)
			.setDstSubpass(0)
			// Source
			.setSrcStageMask(vk::PipelineStageFlagBits::eBottomOfPipe)
			.setSrcAccessMask(vk::AccessFlagBits::eMemoryRead)
			// Destination
			.setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
			.setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite);

		// This dependency transitions the input attachment from color attachment to shader read
		auto subpass_dependency_0_1 = vk::SubpassDependency{}
			.setSrcSubpass(0)
			.setDstSubpass(1)
			// Source
			.setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
			.setSrcAccessMask(vk::AccessFlagBits::eColorAttachmentWrite)
			// Destination
			.setDstStageMask(vk::PipelineStageFlagBits::eFragmentShader)
			.setDstAccessMask(vk::AccessFlagBits::eShaderRead);

		// Create the render pass
		vk::SubpassDescription subpass_descriptions[] = { subpass_description_pong, subpass_description_final };
		vk::SubpassDependency subpass_dependencies[] = { subpass_dependency_ext_0, subpass_dependency_0_1 };
		auto render_pass_create_info = vk::RenderPassCreateInfo{ {}, 2, all_attachment_descriptions, 2, subpass_descriptions, 2, subpass_dependencies };

		render_pass = device->createRenderPassUnique(render_pass_create_info);
	}

	void initialize_descriptor_set_layout()
	{
		// Set up a single descriptor set layout binding - in the shader, these will look like:
		//
		//		layout(set = 0, binding = 0) uniform sampler2D ...;
		//		layout(set = 0, binding = 1) uniform sampler3D ...;
		//		etc.
		//
		// If there were multiple sets, you would need to create a `vk::DescriptorSetLayoutCreateInfo`
		// struct for each set and ensure that the appropriate `vk::DescriptorSetLayoutBinding` structs
		// were associated with that descriptor set layout
		const uint32_t count = 1;
		vk::DescriptorSetLayoutBinding descriptor_set_layout_bindings[2] = 
		{
			// Combined image sampler (used by first subpass)
			vk::DescriptorSetLayoutBinding{
				0, // Binding #0
				vk::DescriptorType::eCombinedImageSampler,
				count,
				vk::ShaderStageFlagBits::eFragment
			},

			// Input attachment (used by second subpass)
			vk::DescriptorSetLayoutBinding{
				1, // Binding #1
				vk::DescriptorType::eInputAttachment,
				count,
				vk::ShaderStageFlagBits::eFragment
			}
		};

		auto descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo{}
			.setPBindings(descriptor_set_layout_bindings)
			.setBindingCount(2);

		descriptor_set_layout = device->createDescriptorSetLayoutUnique(descriptor_set_layout_create_info);
	}

	void initialize_pipeline_layout()
	{
		// Then, create a pipeline layout
		auto push_constant_range = vk::PushConstantRange{ vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushConstants) };
		auto pipeline_layout_create_info = vk::PipelineLayoutCreateInfo{}
			.setPPushConstantRanges(&push_constant_range)
			.setPushConstantRangeCount(1)
			.setPSetLayouts(&descriptor_set_layout.get())
			.setSetLayoutCount(1);

		// Pipeline layout will be shared across both pipelines
		pipeline_layout = device->createPipelineLayoutUnique(pipeline_layout_create_info);
	}

	void initialize_pipelines()
	{
		// First, load all of the shader modules
		const std::string path_prefix = "";
		auto vs_module_quad = load_spv_into_module(device, path_prefix + "quad.spv");
		auto fs_module_pathtrace = load_spv_into_module(device, path_prefix + "pathtrace.spv");
		auto fs_module_composite = load_spv_into_module(device, path_prefix + "composite.spv");
		LOG_DEBUG("Successfully loaded shader modules");

		// Create all of the structs used for pipeline creation: the only thing different between the two
		// pipelines will be the `vk::PipelineShaderStageCreateInfo` below
		const char* entry_point = "main";
auto vs_stage_create_info = vk::PipelineShaderStageCreateInfo{ {}, vk::ShaderStageFlagBits::eVertex, vs_module_quad.get(), entry_point };
auto fs_stage_create_info = vk::PipelineShaderStageCreateInfo{ {}, vk::ShaderStageFlagBits::eFragment, fs_module_pathtrace.get(), entry_point };
vk::PipelineShaderStageCreateInfo shader_stage_create_infos[] = { vs_stage_create_info, fs_stage_create_info };

auto vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo{};

auto input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo{ {}, vk::PrimitiveTopology::eTriangleList };

const float min_depth = 0.0f;
const float max_depth = 1.0f;
const vk::Viewport viewport{ 0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), min_depth, max_depth };
const vk::Rect2D scissor{ { 0, 0 }, swapchain_extent };
auto viewport_state_create_info = vk::PipelineViewportStateCreateInfo{ {}, 1, &viewport, 1, &scissor };

auto rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo{}
.setFrontFace(vk::FrontFace::eClockwise)
.setCullMode(vk::CullModeFlagBits::eBack)
.setLineWidth(1.0f);

auto multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo{};

auto color_blend_attachment_state = vk::PipelineColorBlendAttachmentState{}
.setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

auto color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo{}
.setPAttachments(&color_blend_attachment_state)
.setAttachmentCount(1);

auto graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo{}
.setPStages(shader_stage_create_infos)
.setStageCount(2)
.setPVertexInputState(&vertex_input_state_create_info)
.setPInputAssemblyState(&input_assembly_create_info)
.setPViewportState(&viewport_state_create_info)
.setPRasterizationState(&rasterization_state_create_info)
.setPMultisampleState(&multisample_state_create_info)
.setPColorBlendState(&color_blend_state_create_info)
.setLayout(pipeline_layout.get())
.setRenderPass(render_pass.get())
.setSubpass(0);

// Create the pipeline for pathtracing
pipeline_pathtrace = device->createGraphicsPipelineUnique({}, graphics_pipeline_create_info);

// Switch the fragment shader module and subpass index
shader_stage_create_infos[1] = vk::PipelineShaderStageCreateInfo{ {}, vk::ShaderStageFlagBits::eFragment, fs_module_composite.get(), entry_point };
graphics_pipeline_create_info.setSubpass(1);
pipeline_composite = device->createGraphicsPipelineUnique({}, graphics_pipeline_create_info);

LOG_DEBUG("Created graphics pipelines");
	}

	void initialize_ping_pong_images()
	{
		// We want to create two floating-point images that can be used as both color attachments (i.e. render targets) and combined image samplers
		auto image_create_info = vk::ImageCreateInfo{}
			.setImageType(vk::ImageType::e2D)
			.setFormat(ping_pong_image_format)
			.setExtent({ width, height, 1 })
			.setMipLevels(1)
			.setArrayLayers(1)
			.setUsage(vk::ImageUsageFlagBits::eColorAttachment |
				vk::ImageUsageFlagBits::eSampled |
				vk::ImageUsageFlagBits::eInputAttachment | /* Required for second subpass */
				vk::ImageUsageFlagBits::eTransferDst /* Required for the clear color command to work */);

		image_a = device->createImageUnique(image_create_info);
		image_b = device->createImageUnique(image_create_info);
		LOG_DEBUG("Created images for ping-ponging");

		// Note that at this point, both images are in `vk::ImageLayout::eUndefined`, which does not support device access - we transition them to
		// `vk::ImageLayout::eGeneral` in `clear_ping_pong_images()`

		const auto subresource_range = vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };

		auto image_view_create_info = vk::ImageViewCreateInfo{}
			.setImage(image_a.get())
			.setViewType(vk::ImageViewType::e2D)
			.setFormat(ping_pong_image_format)
			.setSubresourceRange(subresource_range);

		// The `memoryTypeBits` field is a bitmask that contains one bit set for every supported memory type of the resource - bit `i` is set if and only if the
		// memory type `i` in the `vk::PhysicalDeviceMemoryProperties` structure for the physical device is supported for the resource
		auto memory_requirements_a = device->getImageMemoryRequirements(image_a.get());
		auto memory_requirements_b = device->getImageMemoryRequirements(image_b.get());

		// Allocate memory that is owned by the device
		auto desired_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
		auto memory_properties = physical_device.getMemoryProperties();
		auto chosen_memory_type_index = std::numeric_limits<uint32_t>::max();

		for (size_t i = 0; i < memory_properties.memoryTypeCount; ++i)
		{
			if ((memory_requirements_a.memoryTypeBits & (1 << i)) &&
				(memory_requirements_b.memoryTypeBits & (1 << i)) &&
				(memory_properties.memoryTypes[i].propertyFlags & desired_flags) == desired_flags)
			{
				chosen_memory_type_index = i;
			}
		}
		if (chosen_memory_type_index == std::numeric_limits<uint32_t>::max())
		{
			throw std::runtime_error("Could not find suitable memory type for allocation");
		}
		LOG_DEBUG("Allocated device memory from index: " << chosen_memory_type_index);

		// Allocate the device memory that will be shared between the two images
		auto total_memory_size = memory_requirements_a.size + memory_requirements_b.size;
		auto memory_allocation_info = vk::MemoryAllocateInfo{ total_memory_size, chosen_memory_type_index };
		device_memory_ab = device->allocateMemoryUnique(memory_allocation_info);

		// Bind the device memory to each of the two images, with the appropriate offsets
		const vk::DeviceSize offset = memory_requirements_a.size;
		device->bindImageMemory(image_a.get(), device_memory_ab.get(), 0);
		device->bindImageMemory(image_b.get(), device_memory_ab.get(), offset);

		// Create two image views
		image_view_a = device->createImageViewUnique(image_view_create_info);
		image_view_create_info.setImage(image_b.get());
		image_view_b = device->createImageViewUnique(image_view_create_info);
	}

	void initialize_framebuffers()
	{
		ping_pong_framebuffers.clear();

		const uint32_t framebuffer_layers = 1;
		for (size_t i = 0; i < swapchain_image_views.size(); ++i)
		{
			// For each swapchain image, we actually need two framebuffers 
			vk::ImageView attachments_ab[2] = { image_view_a.get(), swapchain_image_views[i].get() }; // AB
			vk::ImageView attachments_ba[2] = { image_view_b.get(), swapchain_image_views[i].get() }; // BA

			auto framebuffer_create_info_ab = vk::FramebufferCreateInfo{ {}, render_pass.get(), 2, attachments_ab, width, height, framebuffer_layers }; // Render into A first, then the swapchain image
			auto framebuffer_create_info_ba = vk::FramebufferCreateInfo{ {}, render_pass.get(), 2, attachments_ba, width, height, framebuffer_layers }; // Render into B first, then the swapchain image

			ping_pong_framebuffers.push_back(device->createFramebufferUnique(framebuffer_create_info_ab)); // AB
			ping_pong_framebuffers.push_back(device->createFramebufferUnique(framebuffer_create_info_ba)); // BA
		}
		LOG_DEBUG("Created [ " << ping_pong_framebuffers.size() << " ] framebuffers");
	}

	void initialize_command_pool()
	{
		// Create the command pool with two flags. The first indicates that command buffers allocated from this pool may be
		// reset individually. Without it, we can't re-record the same command buffer multiple times. The second tells the
		// driver that command buffers allocated from this pool will be re-recorded frequently. This helps optimize command
		// buffer allocation.
		command_pool = device->createCommandPoolUnique(vk::CommandPoolCreateInfo{ 
			vk::CommandPoolCreateFlagBits::eResetCommandBuffer | 
			vk::CommandPoolCreateFlagBits::eTransient, 
			queue_family_index 
		});
	}

	void initialize_command_buffers()
	{
		// We only need one command buffer per pair of framebuffers (AB - BA), thus we divide the length of `ping_pong_framebuffers` by 2
		command_buffers = device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{ command_pool.get(), vk::CommandBufferLevel::ePrimary, static_cast<uint32_t>(ping_pong_framebuffers.size() / 2) });
		LOG_DEBUG("Allocated [ " << command_buffers.size() << " ] command buffers");
	}

	void record_command_buffer(uint32_t index)
	{
		const vk::Rect2D render_area{ { 0, 0 }, swapchain_extent };

		double xpos;
		double ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

		const vk::ClearValue clear_values[] = {
			std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f }, // TODO: is this value needed? 
			std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f }
		};

		int frame_offset = (total_frames_elapsed % 2 == 0) ? 0: 1;

		command_buffers[index]->begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse });
		command_buffers[index]->beginRenderPass(vk::RenderPassBeginInfo{ render_pass.get(), ping_pong_framebuffers[index * 2 + frame_offset].get(), render_area, 2, clear_values }, vk::SubpassContents::eInline);
	
		// TODO: this does not work if it is a bool?
		float mouse_down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS ? 1.0f : 0.0f;

		if (mouse_down == 1.0f)
		{
			samples_per_pixel = 0;
			ttl_passes = 0;
			startTime = clock();
			prvTime = -1;
			lastSaveTime = 0;
		}

		PushConstants push_constants =
		{
			get_elapsed_time(),
			static_cast<float>(samples_per_pixel),
			static_cast<float>(width),
			static_cast<float>(height),
			cursor_position[0],
			cursor_position[1],
			mouse_down
		};

		command_buffers[index]->pushConstants(pipeline_layout.get(), vk::ShaderStageFlagBits::eFragment, 0, sizeof(push_constants), &push_constants);

		if (frame_offset == 0)
		{
			// Bind set A
			command_buffers[index]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout.get(), 0, descriptor_set_a.get(), {});
		}
		else
		{
			// Bind set B
			command_buffers[index]->bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout.get(), 0, descriptor_set_b.get(), {});
		}

		// Bind the pathtracer pipeline object and render a full-screen quad
		command_buffers[index]->bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline_pathtrace.get());
		command_buffers[index]->draw(6, 1, 0, 0);

		// Begin the second subpass
		command_buffers[index]->nextSubpass({});

		// Bind the composite pipeline object and render another full-screen quad
		command_buffers[index]->bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline_composite.get());
		command_buffers[index]->draw(6, 1, 0, 0);

		// End the renderpass
		command_buffers[index]->endRenderPass();
		command_buffers[index]->end();
	}

	void initialize_synchronization_primitives()
	{
		semaphore_image_available = device->createSemaphoreUnique({});
		sempahore_render_finished = device->createSemaphoreUnique({});

		for (size_t i = 0; i < command_buffers.size(); ++i)
		{
			// Create each fence in a signaled state, so that the first call to `waitForFences` in the draw loop doesn't throw any errors
			fences.push_back(device->createFenceUnique({ vk::FenceCreateFlagBits::eSignaled }));
		}
	}

	void one_time_commands(std::function<void(vk::UniqueCommandBuffer const &)> func)
	{
		auto command_buffer = std::move(device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo{ command_pool.get(), vk::CommandBufferLevel::ePrimary, 1 })[0]);

		// Record user-defined commands in between begin/end
		command_buffer->begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eSimultaneousUse });
		func(command_buffer);
		command_buffer->end();

		auto submit_info = vk::SubmitInfo{}
			.setPCommandBuffers(&command_buffer.get())
			.setCommandBufferCount(1);

		// One time submit, so wait idle here for the work to complete
		queue.submit(submit_info, {});
		queue.waitIdle();
	}

	void clear_ping_pong_images()
	{
		const vk::ClearColorValue black = std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 1.0f };
		const auto subresource_range = vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };

		// The memory barrier that will be used to transition each image from `vk::ImageLayout::eUndefined` to `vk::ImageLayout::eGeneral`,
		// which, per the spec, is required before attempting to clear the image
		auto image_memory_barrier = vk::ImageMemoryBarrier{}
			.setSrcAccessMask({})
			.setDstAccessMask({})
			.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
			.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
			.setOldLayout(vk::ImageLayout::eUndefined)
			.setNewLayout(vk::ImageLayout::eGeneral)
			.setSubresourceRange(subresource_range)
			.setImage(image_a.get());

		// Clear both A and B
		one_time_commands([&](const auto& command_buffer) {
			command_buffer->pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTopOfPipe, {}, {}, {}, image_memory_barrier);
			image_memory_barrier.setImage(image_b.get());
			command_buffer->pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTopOfPipe, {}, {}, {}, image_memory_barrier);

			// Both images are now in `vk::ImageLayout::eGeneral`, due to the pipeline barrier above
			command_buffer->clearColorImage(image_a.get(), vk::ImageLayout::eGeneral, black, subresource_range);
			command_buffer->clearColorImage(image_b.get(), vk::ImageLayout::eGeneral, black, subresource_range);
		});

		LOG_DEBUG("Cleared images");
	}

	void initialize_descriptor_pool()
	{
		vk::DescriptorPoolSize descriptor_pool_sizes[2] = {
			vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 2 },
			vk::DescriptorPoolSize{ vk::DescriptorType::eInputAttachment, 2 }
		};
		
		// We need a unique descriptor set for each ping-pong image, although they will share the same layout
		const uint32_t max_sets = 2;
		auto descriptor_pool_create_info = vk::DescriptorPoolCreateInfo{ 
			vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, /* Needed during swapchain resize (which subsequently re-allocates descriptor sets) */
			max_sets, 
			2, descriptor_pool_sizes 
		};
		
		descriptor_pool = device->createDescriptorPoolUnique(descriptor_pool_create_info);
		LOG_DEBUG("Created descriptor pool with [ " << max_sets << " ] possible allocations");
	}

	void initialize_sampler()
	{
		auto sampler_create_info = vk::SamplerCreateInfo{}
			.setMinFilter(vk::Filter::eLinear)
			.setMagFilter(vk::Filter::eLinear);

		sampler = device->createSamplerUnique(sampler_create_info);
	}

	void initialize_descriptor_sets()
	{
		// We allocate descriptors from the descriptor pool created above
		auto descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo{ descriptor_pool.get(), 1, &descriptor_set_layout.get() };

		// In this project, we actually use two separate (but identical in terms of # of descriptors, types, etc.) - this is because
		// the image that serves as our "back-buffer" during the ping-ponging changes from frame to frame
		descriptor_set_a = std::move(device->allocateDescriptorSetsUnique(descriptor_set_allocate_info)[0]);
		descriptor_set_b = std::move(device->allocateDescriptorSetsUnique(descriptor_set_allocate_info)[0]);
		LOG_DEBUG("Allocated descriptor set(s)");

		// These are hard-coded bindings in the shader and therefore do not change between A / B
		const uint32_t cis_binding = 0;
		const uint32_t input_binding = 1;

		auto descriptor_image_info_a = vk::DescriptorImageInfo{ sampler.get(), image_view_a.get(), vk::ImageLayout::eShaderReadOnlyOptimal }; // TODO: no idea what these layouts should be...
		auto descriptor_image_info_b = vk::DescriptorImageInfo{ sampler.get(), image_view_b.get(), vk::ImageLayout::eShaderReadOnlyOptimal };

		// Write to descriptor set A, where A is the intermediate render target
		{
			auto write_descriptor_set_cis = vk::WriteDescriptorSet{}
				.setDescriptorCount(1)
				.setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
				.setDstBinding(cis_binding)
				.setDstSet(descriptor_set_a.get())
				.setPImageInfo(&descriptor_image_info_b);

			auto write_descriptor_set_input = vk::WriteDescriptorSet{}
				.setDescriptorCount(1)
				.setDescriptorType(vk::DescriptorType::eInputAttachment)
				.setDstBinding(input_binding)
				.setDstSet(descriptor_set_a.get())
				.setPImageInfo(&descriptor_image_info_a);

			device->updateDescriptorSets({ write_descriptor_set_cis, write_descriptor_set_input }, {});
		}

		// Write to descriptor set B, where B is the intermediate render target
		{
			auto write_descriptor_set_cis = vk::WriteDescriptorSet{}
				.setDescriptorCount(1)
				.setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
				.setDstBinding(cis_binding)
				.setDstSet(descriptor_set_b.get())
				.setPImageInfo(&descriptor_image_info_a);

			auto write_descriptor_set_input = vk::WriteDescriptorSet{}
				.setDescriptorCount(1)
				.setDescriptorType(vk::DescriptorType::eInputAttachment)
				.setDstBinding(input_binding)
				.setDstSet(descriptor_set_b.get())
				.setPImageInfo(&descriptor_image_info_b);

			device->updateDescriptorSets({ write_descriptor_set_cis, write_descriptor_set_input }, {});
		}

		LOG_DEBUG("Wrote descriptor set(s)");
	}

	void draw()
	{
		while (!glfwWindowShouldClose(window)) 
		{
			glfwPollEvents();

			// Submit a command buffer after acquiring the index of the next available swapchain image
			auto index = device->acquireNextImageKHR(swapchain.get(), (std::numeric_limits<uint64_t>::max)(), semaphore_image_available.get(), {}).value;

			// If the command buffer we want to (re)use is still pending on the GPU, wait for it then reset its fence
			device->waitForFences(fences[index].get(), true, (std::numeric_limits<uint64_t>::max)());
			device->resetFences(fences[index].get());

			// Now, we know that we can safely (re)use this command buffer
			record_command_buffer(index);
			
			const vk::PipelineStageFlags wait_stages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
			auto submit_info = vk::SubmitInfo{ 1, &semaphore_image_available.get(), wait_stages, 1, &command_buffers[index].get(), 1, &sempahore_render_finished.get() };
			queue.submit(submit_info, fences[index].get());

			// Present the final rendered image to the swapchain
			auto present_info = vk::PresentInfoKHR{ 1, &sempahore_render_finished.get(), 1, &swapchain.get(), &index };
			queue.presentKHR(present_info);


			//tigra: samples count
			samples_per_pixel++;
			total_frames_elapsed++;
			
			ttl_passes++;
			
			
			timeNow = clock();
			
			timeSecs = float(timeNow - startTime) / float(CLOCKS_PER_SEC);
			
			
			if(saveEveryTime > 0)
			{
				if(timeSecs - lastSaveTime > float(saveEveryTime)+0.001f)
				{
					instantSave();
					lastSaveTime = timeSecs;
				}
			}
			
			if(maxTime > 0)
			{
				if(timeSecs - maxTime > 0.001f)
				{
					instantSave();
					exit(1);
				}
			}
			
			if(saveEveryPasses > 0)
			{
				if(ttl_passes > 0 && (ttl_passes % saveEveryPasses == 0) )
				{
					instantSave();
					lastSaveTime = timeSecs;
				}
			}
			
			if(maxPasses > 0)
			{
				if(ttl_passes > 0 && ttl_passes==maxPasses)
				{
					instantSave();
					exit(1);
				}
			}
			
			if(timeSecs - prvTime > 0.45f)
			{
				prvTime = timeSecs;
				
				snprintf(strrr, 1000, "VULKANized pepelac  %ld  %.1fs   %.2f pass/s",
								samples_per_pixel,
								timeSecs,
								float(samples_per_pixel) / timeSecs
						);
						
				glfwSetWindowTitle (window, strrr);
			}
		}
	}

private:
	uint32_t width;
	uint32_t height;
	std::string name;
	uint32_t samples_per_pixel;
	uint32_t total_frames_elapsed;
	float cursor_position[2];

	GLFWwindow* window;

	vk::SurfaceCapabilitiesKHR surface_capabilities;
	std::vector<vk::SurfaceFormatKHR> surface_formats;
	std::vector<vk::PresentModeKHR> surface_present_modes;

	vk::Format ping_pong_image_format;
	vk::Format swapchain_image_format;

	vk::Extent2D swapchain_extent;

	vk::PhysicalDevice physical_device;
	vk::DebugReportCallbackEXT debug_report_callback;
	vk::Queue queue;
	uint32_t queue_family_index;

	vk::UniqueInstance instance;
	vk::UniqueDevice device;
	vk::UniqueSurfaceKHR surface;
	vk::UniqueSwapchainKHR swapchain;
	vk::UniqueRenderPass render_pass;
	vk::UniquePipelineLayout pipeline_layout;
	vk::UniquePipeline pipeline_pathtrace;
	vk::UniquePipeline pipeline_composite;
	vk::UniqueCommandPool command_pool;
	vk::UniqueSemaphore semaphore_image_available;
	vk::UniqueSemaphore sempahore_render_finished;
	std::vector<vk::Image> swapchain_images;
	std::vector<vk::UniqueImageView> swapchain_image_views;

	std::vector<vk::UniqueCommandBuffer> command_buffers;
	std::vector<vk::UniqueFence> fences;

	vk::UniqueImage image_a;
	vk::UniqueImage image_b;
	vk::UniqueDeviceMemory device_memory_ab;
	vk::UniqueImageView image_view_a;
	vk::UniqueImageView image_view_b;

	vk::UniqueDescriptorSetLayout descriptor_set_layout;
	vk::UniqueDescriptorPool descriptor_pool;
	vk::UniqueDescriptorSet descriptor_set_a;
	vk::UniqueDescriptorSet descriptor_set_b;
	vk::UniqueSampler sampler;

	// Least common multiple of the number of swapchain images (3) and number of ping-pong buffers (2)
	std::vector<vk::UniqueFramebuffer> ping_pong_framebuffers;
};



void processCommandLine(int argc, char *argv[])
{
	int i, tmp;
	
	for(i=1;i<argc;i++)
	{
		if(strcmp(argv[i], "-t")==0)
		{
			if(i+1<argc)
			{
				tmp = atoi(argv[i+1]);
				if(tmp>0)
				{
					maxTime = tmp;
					printf(":maxTime: %ds\n", maxTime);
				}
				i++;
			}
		}
		else
		if(strcmp(argv[i], "-p")==0)
		{
			if(i+1<argc)
			{
				tmp = atoi(argv[i+1]);
				if(tmp>0)
				{
					maxPasses = tmp;
					printf(":maxPasses: %d\n", maxPasses);
				}
				i++;
			}
		}
		else
		if(strcmp(argv[i], "-et")==0)
		{
			if(i+1<argc)
			{
				tmp = atoi(argv[i+1]);
				if(tmp>0)
				{
					saveEveryTime = tmp;
					printf(":saveEveryTime: %ds\n", saveEveryTime);
				}
				i++;
			}
		}
		else
		if(strcmp(argv[i], "-ep")==0)
		{
			if(i+1<argc)
			{
				tmp = atoi(argv[i+1]);
				if(tmp>0)
				{
					saveEveryPasses = tmp;
					printf(":saveEveryPasses: %d\n", saveEveryPasses);
				}
				i++;
			}
		}
		else
		if(strcmp(argv[i], "--prefix")==0)
		{
			if(i+1<argc)
			{
				fn_prefix = argv[i+1];
				
					printf(":--prefix: %s\n", fn_prefix);
					
				i++;
			}
		}
	}
}



int main(int argc, char *argv[])
{	
	processCommandLine(argc, argv);
	
	Application app{ 800, 800, "flow" };
	app.draw();

	return EXIT_SUCCESS;
}
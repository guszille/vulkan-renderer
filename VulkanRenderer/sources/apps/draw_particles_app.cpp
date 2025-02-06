#include "draw_particles_app.h"

DrawParticlesApp::DrawParticlesApp()
{
}

void DrawParticlesApp::setup(GLFWwindow* window)
{
	logExtensionSupport();

	createInstance();
	createDebugMessenger();
	createSurface(window);

	selectPhysicalDevice();
	createLogicalDevice();

	createSwapChain(window);
	createImageViews();

	createRenderPass();
	createDescriptorSetLayout();
	createGraphicsPipeline();
	createComputePipeline();

	createCommandPool();
	createGraphicsCommandBuffers();
	createComputeCommandBuffers();

	createColorResources();
	createDepthResources();

	createFramebuffers();

	createShaderStorageBuffers();
	createUniformBuffers();

	createDescriptorPool();
	createDescriptorSets();

	createSyncObjects();
}

void DrawParticlesApp::cleanUp()
{
	vkDeviceWaitIdle(context.device);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vkDestroyBuffer(context.device, context.shaderStorageBuffers[i], nullptr);
		vkFreeMemory(context.device, context.shaderStorageBuffersMemory[i], nullptr);
	}

	for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vkDestroyBuffer(context.device, context.uniformBuffers[i], nullptr);
		vkFreeMemory(context.device, context.uniformBuffersMemory[i], nullptr);
	}

	if (ENABLE_VALIDATION_LAYERS)
	{
		destroyDebugUtilsMessengerEXT(context.instance, context.debugMessenger, nullptr);
	}

	for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		vkDestroySemaphore(context.device, context.swapChainAcquireSemaphores[i], nullptr);
		vkDestroySemaphore(context.device, context.swapChainReleaseSemaphores[i], nullptr);
		vkDestroyFence(context.device, context.graphicsSubmitFences[i], nullptr);

		vkDestroySemaphore(context.device, context.computeFinishedSemaphores[i], nullptr);
		vkDestroyFence(context.device, context.computeSubmitFences[i], nullptr);
	}

	vkDestroyCommandPool(context.device, context.commandPool, nullptr);

	for (uint32_t i = 0; i < context.swapChainFramebuffers.size(); i++)
	{
		vkDestroyFramebuffer(context.device, context.swapChainFramebuffers[i], nullptr);
	}

	vkDestroyImageView(context.device, context.depthImageView, nullptr);
	vkDestroyImage(context.device, context.depthImage, nullptr);
	vkFreeMemory(context.device, context.depthImageMemory, nullptr);

	vkDestroyImageView(context.device, context.colorImageView, nullptr);
	vkDestroyImage(context.device, context.colorImage, nullptr);
	vkFreeMemory(context.device, context.colorImageMemory, nullptr);

	vkDestroyPipeline(context.device, context.graphicsPipeline, nullptr);
	vkDestroyPipeline(context.device, context.computePipeline, nullptr);

	vkDestroyPipelineLayout(context.device, context.graphicsPipelineLayout, nullptr);
	vkDestroyPipelineLayout(context.device, context.computePipelineLayout, nullptr);

	vkDestroyDescriptorPool(context.device, context.descriptorPool, nullptr);

	vkDestroyDescriptorSetLayout(context.device, context.descriptorSetLayout, nullptr);

	vkDestroyRenderPass(context.device, context.renderPass, nullptr);

	for (uint32_t i = 0; i < context.swapChainImageViews.size(); i++)
	{
		vkDestroyImageView(context.device, context.swapChainImageViews[i], nullptr);
	}

	vkDestroySwapchainKHR(context.device, context.swapChain, nullptr);

	vkDestroyDevice(context.device, nullptr);

	vkDestroySurfaceKHR(context.instance, context.surface, nullptr);

	vkDestroyInstance(context.instance, nullptr);
}

void DrawParticlesApp::update(float deltaTime)
{
	context.currentTime += deltaTime;
}

void DrawParticlesApp::render(GLFWwindow* window, float deltaTime)
{
	// Compute submission.
	vkWaitForFences(context.device, 1, &context.computeSubmitFences[context.currentFrame], VK_TRUE, UINT64_MAX);

	updateUniformBuffer(context.currentFrame);

	vkResetFences(context.device, 1, &context.computeSubmitFences[context.currentFrame]);

	vkResetCommandBuffer(context.computeCommandBuffers[context.currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
	
	recordComputeCommandBuffer(context.computeCommandBuffers[context.currentFrame]);

	VkSubmitInfo computeSubmitInfo{};

	computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	computeSubmitInfo.commandBufferCount = 1;
	computeSubmitInfo.pCommandBuffers = &context.computeCommandBuffers[context.currentFrame];
	computeSubmitInfo.signalSemaphoreCount = 1;
	computeSubmitInfo.pSignalSemaphores = &context.computeFinishedSemaphores[context.currentFrame];

	if (vkQueueSubmit(context.computeQueue, 1, &computeSubmitInfo, context.computeSubmitFences[context.currentFrame]) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to submit compute command buffer!");
	}

	// Graphics submission.
	vkWaitForFences(context.device, 1, &context.graphicsSubmitFences[context.currentFrame], VK_TRUE, UINT64_MAX);

	uint32_t imageIndex;
	VkResult acquireResult = vkAcquireNextImageKHR(context.device, context.swapChain, UINT64_MAX, context.swapChainAcquireSemaphores[context.currentFrame], VK_NULL_HANDLE, &imageIndex);

	if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR || acquireResult == VK_SUBOPTIMAL_KHR)
	{
		recreateSwapChain(window);
		return;
	}
	else if (acquireResult != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to acquire swap chain image!");
	}

	vkResetFences(context.device, 1, &context.graphicsSubmitFences[context.currentFrame]); // Only reset the fence if we are submitting some work...

	vkResetCommandBuffer(context.graphicsCommandBuffers[context.currentFrame], 0);

	recordGraphicsCommandBuffer(context.graphicsCommandBuffers[context.currentFrame], imageIndex);

	VkSemaphore waitSemaphores[] = { context.computeFinishedSemaphores[context.currentFrame], context.swapChainAcquireSemaphores[context.currentFrame] };
	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

	VkSubmitInfo graphicsSubmitInfo{};

	graphicsSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	graphicsSubmitInfo.waitSemaphoreCount = 2; // Attention to this number.
	graphicsSubmitInfo.pWaitSemaphores = waitSemaphores;
	graphicsSubmitInfo.pWaitDstStageMask = waitStages;
	graphicsSubmitInfo.commandBufferCount = 1;
	graphicsSubmitInfo.pCommandBuffers = &context.graphicsCommandBuffers[context.currentFrame];
	graphicsSubmitInfo.signalSemaphoreCount = 1;
	graphicsSubmitInfo.pSignalSemaphores = &context.swapChainReleaseSemaphores[context.currentFrame];

	if (vkQueueSubmit(context.graphicsQueue, 1, &graphicsSubmitInfo, context.graphicsSubmitFences[context.currentFrame]) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to submit draw command buffer!");
	}

	VkSwapchainKHR swapChains[] = { context.swapChain };

	VkPresentInfoKHR presentInfo{};

	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &context.swapChainReleaseSemaphores[context.currentFrame];
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = swapChains;
	presentInfo.pImageIndices = &imageIndex;
	presentInfo.pResults = nullptr;

	VkResult presentResult = vkQueuePresentKHR(context.presentQueue, &presentInfo);

	if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR || framebufferResized)
	{
		recreateSwapChain(window);

		framebufferResized = false;
	}
	else if (presentResult != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to present swap chain image!");
	}

	context.currentFrame = (context.currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void DrawParticlesApp::logExtensionSupport()
{
	uint32_t extensionCount = 0;

	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

	std::vector<VkExtensionProperties> availableExtensions(extensionCount);

	vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data());

	std::cout << "[INFO] AVAILABLE EXTENSIONS:" << std::endl;

	for (const VkExtensionProperties& extension : availableExtensions)
	{
		std::cout << '\t' << extension.extensionName << std::endl;
	}
}

bool DrawParticlesApp::checkValidationLayerSupport()
{
	uint32_t layerCount = 0;

	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

	std::vector<VkLayerProperties> availableLayers(layerCount);

	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

	for (const char* layerName : VALIDATION_LAYERS)
	{
		bool layerFound = false;

		for (const VkLayerProperties& layerProperties : availableLayers)
		{
			if (strcmp(layerName, layerProperties.layerName) == 0)
			{
				layerFound = true;
				break;
			}
		}

		if (!layerFound)
		{
			return false;
		}
	}

	return true;
}

std::vector<const char*> DrawParticlesApp::getRequiredInstanceExtensions()
{
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

	if (ENABLE_VALIDATION_LAYERS)
	{
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	return extensions;
}

bool DrawParticlesApp::checkDeviceExtensionSupport(VkPhysicalDevice device)
{
	uint32_t extensionCount = 0;

	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

	std::vector<VkExtensionProperties> availableExtensions(extensionCount);

	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

	for (const char* extensionName : DEVICE_EXTENSIONS)
	{
		bool extensionFound = false;

		for (const auto& extensionProperties : availableExtensions)
		{
			if (strcmp(extensionName, extensionProperties.extensionName) == 0)
			{
				extensionFound = true;
				break;
			}
		}

		if (!extensionFound)
		{
			return false;
		}
	}

	return true;
}

QueueFamilyIndices DrawParticlesApp::findQueueFamilies(VkPhysicalDevice device)
{
	uint32_t queueFamilyCount = 0;

	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);

	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

	QueueFamilyIndices indices;

	for (uint32_t i = 0; i < queueFamilyCount; i++)
	{
		const VkQueueFamilyProperties& queueFamily = queueFamilies[i];
		VkBool32 presentSupport = false;

		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, context.surface, &presentSupport);

		if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT))
		{
			indices.graphicsAndComputeFamily = i;
		}

		if (presentSupport)
		{
			indices.presentFamily = i;
		}

		if (indices.isComplete())
		{
			break;
		}
	}

	return indices;
}

SwapChainSupportDetails DrawParticlesApp::querySwapChainSupport(VkPhysicalDevice device)
{
	SwapChainSupportDetails details;

	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, context.surface, &details.capabilities);

	uint32_t formatCount = 0;

	vkGetPhysicalDeviceSurfaceFormatsKHR(device, context.surface, &formatCount, nullptr);

	if (formatCount != 0)
	{
		details.formats.resize(formatCount);

		vkGetPhysicalDeviceSurfaceFormatsKHR(device, context.surface, &formatCount, details.formats.data());
	}

	uint32_t presentModeCount = 0;

	vkGetPhysicalDeviceSurfacePresentModesKHR(device, context.surface, &presentModeCount, nullptr);

	if (presentModeCount != 0)
	{
		details.presentModes.resize(presentModeCount);

		vkGetPhysicalDeviceSurfacePresentModesKHR(device, context.surface, &presentModeCount, details.presentModes.data());
	}

	return details;
}

bool DrawParticlesApp::isDeviceSuitable(VkPhysicalDevice device)
{
	VkPhysicalDeviceProperties deviceProperties{};

	vkGetPhysicalDeviceProperties(device, &deviceProperties);

	VkPhysicalDeviceFeatures deviceFeatures{};

	vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

	QueueFamilyIndices queueFamilyIndices = findQueueFamilies(device);
	bool extensionsSupported = checkDeviceExtensionSupport(device);
	bool swapChainAdequate = false;

	if (extensionsSupported)
	{
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);

		swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
	}

	bool suitable = true;

	suitable = suitable && deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
	suitable = suitable && deviceFeatures.geometryShader && deviceFeatures.samplerAnisotropy;
	suitable = suitable && queueFamilyIndices.isComplete();
	suitable = suitable && extensionsSupported && swapChainAdequate;

	return suitable;
}

VkSampleCountFlagBits DrawParticlesApp::getMaxUsableSampleCount()
{
	VkPhysicalDeviceProperties deviceProperties{};

	vkGetPhysicalDeviceProperties(context.gpu, &deviceProperties);

	VkSampleCountFlags counts = deviceProperties.limits.framebufferColorSampleCounts & deviceProperties.limits.framebufferDepthSampleCounts;

	if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
	if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
	if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
	if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
	if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
	if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

	return VK_SAMPLE_COUNT_1_BIT;
}

VkSurfaceFormatKHR DrawParticlesApp::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
	for (const VkSurfaceFormatKHR& availableFormat : availableFormats)
	{
		if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
		{
			return availableFormat;
		}
	}

	return availableFormats[0];
}

VkPresentModeKHR DrawParticlesApp::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
{
	for (const VkPresentModeKHR& availablePresentMode : availablePresentModes)
	{
		if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
		{
			return availablePresentMode;
		}
	}

	return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D DrawParticlesApp::chooseSwapExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities)
{
	if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
	{
		return capabilities.currentExtent;
	}
	else
	{
		int width, height;

		glfwGetFramebufferSize(window, &width, &height);

		VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

		actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

		return actualExtent;
	}
}

VkImageView DrawParticlesApp::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels)
{
	VkImageViewCreateInfo viewCreateInfo{};

	viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewCreateInfo.image = image;
	viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewCreateInfo.format = format;
	viewCreateInfo.subresourceRange.aspectMask = aspectFlags;
	viewCreateInfo.subresourceRange.baseMipLevel = 0;
	viewCreateInfo.subresourceRange.levelCount = mipLevels;
	viewCreateInfo.subresourceRange.baseArrayLayer = 0;
	viewCreateInfo.subresourceRange.layerCount = 1;

	VkImageView imageView{};

	if (vkCreateImageView(context.device, &viewCreateInfo, nullptr, &imageView) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create texture image view!");
	}

	return imageView;
}

VkShaderModule DrawParticlesApp::createShaderModule(const std::vector<char>& code)
{
	VkShaderModuleCreateInfo shaderModuleCreateInfo{};

	shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	shaderModuleCreateInfo.codeSize = code.size();
	shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

	VkShaderModule shaderModule{};

	if (vkCreateShaderModule(context.device, &shaderModuleCreateInfo, nullptr, &shaderModule) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create shader module!");
	}

	return shaderModule;
}

VkFormat DrawParticlesApp::findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
{
	for (const VkFormat& format : candidates)
	{
		VkFormatProperties properties{};

		vkGetPhysicalDeviceFormatProperties(context.gpu, format, &properties);

		if (tiling == VK_IMAGE_TILING_LINEAR && (properties.linearTilingFeatures & features) == features)
		{
			return format;
		}

		if (tiling == VK_IMAGE_TILING_OPTIMAL && (properties.optimalTilingFeatures & features) == features)
		{
			return format;
		}
	}

	throw std::runtime_error("Failed to find supported format!");
}

VkFormat DrawParticlesApp::findDepthFormat()
{
	return findSupportedFormat({ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT }, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

bool DrawParticlesApp::hasStencilComponent(VkFormat format)
{
	return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

void DrawParticlesApp::cleanUpSwapChain()
{
	for (uint32_t i = 0; i < context.swapChainFramebuffers.size(); i++)
	{
		vkDestroyFramebuffer(context.device, context.swapChainFramebuffers[i], nullptr);
	}

	for (uint32_t i = 0; i < context.swapChainImageViews.size(); i++)
	{
		vkDestroyImageView(context.device, context.swapChainImageViews[i], nullptr);
	}

	vkDestroySwapchainKHR(context.device, context.swapChain, nullptr);
}

void DrawParticlesApp::recreateSwapChain(GLFWwindow* window)
{
	int width = 0, height = 0;

	do
	{
		glfwGetFramebufferSize(window, &width, &height);
		glfwWaitEvents();
	} while (width == 0 || height == 0);

	vkDeviceWaitIdle(context.device);

	cleanUpSwapChain();

	createSwapChain(window);
	createImageViews();

	createColorResources();
	createDepthResources();

	createFramebuffers();
}

void DrawParticlesApp::recordGraphicsCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
{
	VkCommandBufferBeginInfo commandBufferBeginInfo{};

	commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	commandBufferBeginInfo.flags = 0;
	commandBufferBeginInfo.pInheritanceInfo = nullptr;

	if (vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to begin recording graphics command buffer!");
	}

	std::array<VkClearValue, 2> clearValues{}; // The order of "clearValues" should be identical to the order of your attachments.

	clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
	clearValues[1].depthStencil = { 1.0f, 0 };

	VkRenderPassBeginInfo renderPassBeginInfo{};

	renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassBeginInfo.renderPass = context.renderPass;
	renderPassBeginInfo.framebuffer = context.swapChainFramebuffers[imageIndex];
	renderPassBeginInfo.renderArea.offset = { 0, 0 };
	renderPassBeginInfo.renderArea.extent = context.swapChainExtent;
	renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
	renderPassBeginInfo.pClearValues = clearValues.data();

	vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, context.graphicsPipeline);

	VkViewport viewport{};

	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = static_cast<float>(context.swapChainExtent.width);
	viewport.height = static_cast<float>(context.swapChainExtent.height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

	VkRect2D scissor{};

	scissor.offset = { 0, 0 };
	scissor.extent = context.swapChainExtent;

	vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

	VkDeviceSize offsets[] = { 0 };

	vkCmdBindVertexBuffers(commandBuffer, 0, 1, &context.shaderStorageBuffers[context.currentFrame], offsets);

	vkCmdDraw(commandBuffer, particleCount, 1, 0, 0);

	vkCmdEndRenderPass(commandBuffer);

	if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to record graphics command buffer!");
	}
}

void DrawParticlesApp::recordComputeCommandBuffer(VkCommandBuffer commandBuffer)
{
	VkCommandBufferBeginInfo commandBufferBeginInfo{};

	commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	commandBufferBeginInfo.flags = 0;
	commandBufferBeginInfo.pInheritanceInfo = nullptr;

	if (vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to begin recording compute command buffer!");
	}

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, context.computePipeline);

	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, context.computePipelineLayout, 0, 1, &context.descriptorSets[context.currentFrame], 0, nullptr);

	vkCmdDispatch(commandBuffer, particleCount / 256, 1, 1);

	if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to record compute command buffer!");
	}
}

VkCommandBuffer DrawParticlesApp::beginSingleTimeCommands()
{
	VkCommandBufferAllocateInfo commandBufferAllocateInfo{};

	commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandPool = context.commandPool;
	commandBufferAllocateInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;

	vkAllocateCommandBuffers(context.device, &commandBufferAllocateInfo, &commandBuffer);

	VkCommandBufferBeginInfo beginInfo{};

	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	return commandBuffer;
}

void DrawParticlesApp::endSingleTimeCommands(VkCommandBuffer commandBuffer)
{
	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo{};

	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	vkQueueSubmit(context.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);

	vkQueueWaitIdle(context.graphicsQueue);

	vkFreeCommandBuffers(context.device, context.commandPool, 1, &commandBuffer);
}

uint32_t DrawParticlesApp::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
	VkPhysicalDeviceMemoryProperties memoryProperties{};

	vkGetPhysicalDeviceMemoryProperties(context.gpu, &memoryProperties);

	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
	{
		if (typeFilter & (1 << i) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
		{
			return i;
		}
	}

	throw std::runtime_error("Failed to find suitable memory type!");
}

void DrawParticlesApp::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
{
	VkBufferCreateInfo bufferCreateInfo{};

	bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferCreateInfo.size = size;
	bufferCreateInfo.usage = usage;
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	if (vkCreateBuffer(context.device, &bufferCreateInfo, nullptr, &buffer) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create buffer!");
	}

	VkMemoryRequirements memoryRequirements{};

	vkGetBufferMemoryRequirements(context.device, buffer, &memoryRequirements);

	VkMemoryAllocateInfo memoryAllocateInfo{};

	memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memoryAllocateInfo.allocationSize = memoryRequirements.size;
	memoryAllocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, properties);

	if (vkAllocateMemory(context.device, &memoryAllocateInfo, nullptr, &bufferMemory) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate buffer memory!");
	}

	vkBindBufferMemory(context.device, buffer, bufferMemory, 0);
}

void DrawParticlesApp::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	VkBufferCopy copyRegion{};

	copyRegion.srcOffset = 0;
	copyRegion.dstOffset = 0;
	copyRegion.size = size;

	vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

	endSingleTimeCommands(commandBuffer);
}

void DrawParticlesApp::createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
{
	VkImageCreateInfo imageCreateInfo{};

	imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
	imageCreateInfo.extent.width = width;
	imageCreateInfo.extent.height = height;
	imageCreateInfo.extent.depth = 1;
	imageCreateInfo.mipLevels = mipLevels;
	imageCreateInfo.arrayLayers = 1;
	imageCreateInfo.format = format;
	imageCreateInfo.tiling = tiling;
	imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageCreateInfo.usage = usage;
	imageCreateInfo.samples = numSamples;
	imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	if (vkCreateImage(context.device, &imageCreateInfo, nullptr, &image) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create image!");
	}

	VkMemoryRequirements memoryRequirements{};

	vkGetImageMemoryRequirements(context.device, image, &memoryRequirements);

	VkMemoryAllocateInfo memoryAllocateInfo{};

	memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memoryAllocateInfo.allocationSize = memoryRequirements.size;
	memoryAllocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, properties);

	if (vkAllocateMemory(context.device, &memoryAllocateInfo, nullptr, &imageMemory) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate image memory!");
	}

	vkBindImageMemory(context.device, image, imageMemory, 0);
}

void DrawParticlesApp::updateUniformBuffer(uint32_t currentImage)
{
	UniformBufferObject ubo{};

	ubo.time = context.currentTime * 0.5f;

	memcpy(context.uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

void DrawParticlesApp::createInstance()
{
	if (ENABLE_VALIDATION_LAYERS && !checkValidationLayerSupport())
	{
		throw std::runtime_error("Validation layers requested, but not available!");
	}

	std::vector<const char*> extensions = getRequiredInstanceExtensions();

	VkApplicationInfo appInfo{};

	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "Vulkan Renderer";
	appInfo.pEngineName = "No Engine";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo instanceCreateInfo{};

	instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instanceCreateInfo.pApplicationInfo = &appInfo;
	instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	instanceCreateInfo.ppEnabledExtensionNames = extensions.data();

	if (ENABLE_VALIDATION_LAYERS)
	{
		VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo{};

		debugMessengerCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		debugMessengerCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		debugMessengerCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		debugMessengerCreateInfo.pfnUserCallback = debugCallback;

		instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
		instanceCreateInfo.ppEnabledLayerNames = VALIDATION_LAYERS.data();

		instanceCreateInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugMessengerCreateInfo;
	}
	else
	{
		instanceCreateInfo.enabledLayerCount = 0;

		instanceCreateInfo.pNext = nullptr;
	}

	if (vkCreateInstance(&instanceCreateInfo, nullptr, &context.instance) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create VK instance!");
	}
}

void DrawParticlesApp::createDebugMessenger()
{
	if (ENABLE_VALIDATION_LAYERS)
	{
		VkDebugUtilsMessengerCreateInfoEXT messengerCreateInfo{};

		messengerCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		messengerCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		messengerCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		messengerCreateInfo.pfnUserCallback = debugCallback;

		if (createDebugUtilsMessengerEXT(context.instance, &messengerCreateInfo, nullptr, &context.debugMessenger) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to set up debug messenger!");
		}
	}
}

void DrawParticlesApp::createSurface(GLFWwindow* window)
{
	if (glfwCreateWindowSurface(context.instance, window, nullptr, &context.surface) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create window surface!");
	}
}

void DrawParticlesApp::selectPhysicalDevice()
{
	uint32_t deviceCount = 0;

	vkEnumeratePhysicalDevices(context.instance, &deviceCount, nullptr);

	if (deviceCount == 0)
	{
		throw std::runtime_error("Failed to find GPUs with Vulkan support!");
	}

	std::vector<VkPhysicalDevice> availableDevices(deviceCount);

	vkEnumeratePhysicalDevices(context.instance, &deviceCount, availableDevices.data());

	for (const VkPhysicalDevice& device : availableDevices)
	{
		if (isDeviceSuitable(device))
		{
			context.gpu = device;
			context.msaaSamples = getMaxUsableSampleCount();

			break;
		}
	}

	if (context.gpu == VK_NULL_HANDLE)
	{
		throw std::runtime_error("Failed to find a suitable GPU!");
	}
}

void DrawParticlesApp::createLogicalDevice()
{
	float queuePriority = 1.0f;

	QueueFamilyIndices indices = findQueueFamilies(context.gpu);
	std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsAndComputeFamily.value(), indices.presentFamily.value() };
	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

	for (uint32_t queueFamily : uniqueQueueFamilies)
	{
		VkDeviceQueueCreateInfo queueCreateInfo{};

		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = queueFamily;
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &queuePriority;

		queueCreateInfos.push_back(queueCreateInfo);
	}

	VkPhysicalDeviceFeatures deviceFeatures{};

	deviceFeatures.samplerAnisotropy = VK_TRUE;
	deviceFeatures.sampleRateShading = VK_FALSE; // To enable/disable sample shading feature for the device.

	VkDeviceCreateInfo deviceCreateInfo{};

	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
	deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
	deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
	deviceCreateInfo.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();

	if (ENABLE_VALIDATION_LAYERS)
	{
		deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
		deviceCreateInfo.ppEnabledLayerNames = VALIDATION_LAYERS.data();
	}
	else
	{
		deviceCreateInfo.enabledLayerCount = 0;
	}

	if (vkCreateDevice(context.gpu, &deviceCreateInfo, nullptr, &context.device) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create logical device!");
	}

	vkGetDeviceQueue(context.device, indices.graphicsAndComputeFamily.value(), 0, &context.graphicsQueue);
	vkGetDeviceQueue(context.device, indices.graphicsAndComputeFamily.value(), 0, &context.computeQueue); // Using same index as graphics queue.
	vkGetDeviceQueue(context.device, indices.presentFamily.value(), 0, &context.presentQueue);
}

void DrawParticlesApp::createSwapChain(GLFWwindow* window)
{
	SwapChainSupportDetails swapChainSupport = querySwapChainSupport(context.gpu);

	VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
	VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
	VkExtent2D extent = chooseSwapExtent(window, swapChainSupport.capabilities);

	uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

	if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
	{
		imageCount = swapChainSupport.capabilities.maxImageCount;
	}

	VkSwapchainCreateInfoKHR swapchainCreateInfo{};

	swapchainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	swapchainCreateInfo.surface = context.surface;
	swapchainCreateInfo.minImageCount = imageCount;
	swapchainCreateInfo.imageFormat = surfaceFormat.format;
	swapchainCreateInfo.imageColorSpace = surfaceFormat.colorSpace;
	swapchainCreateInfo.imageExtent = extent;
	swapchainCreateInfo.imageArrayLayers = 1;
	swapchainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	QueueFamilyIndices indices = findQueueFamilies(context.gpu);
	uint32_t queueFamilyIndices[] = { indices.graphicsAndComputeFamily.value(), indices.presentFamily.value() };

	if (indices.graphicsAndComputeFamily != indices.presentFamily)
	{
		swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;

		swapchainCreateInfo.queueFamilyIndexCount = 2;
		swapchainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
	}
	else
	{
		swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;

		swapchainCreateInfo.queueFamilyIndexCount = 0;
		swapchainCreateInfo.pQueueFamilyIndices = nullptr;
	}

	swapchainCreateInfo.preTransform = swapChainSupport.capabilities.currentTransform;
	swapchainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	swapchainCreateInfo.presentMode = presentMode;
	swapchainCreateInfo.clipped = VK_TRUE;
	swapchainCreateInfo.oldSwapchain = VK_NULL_HANDLE;

	if (vkCreateSwapchainKHR(context.device, &swapchainCreateInfo, nullptr, &context.swapChain) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create swap chain!");
	}

	vkGetSwapchainImagesKHR(context.device, context.swapChain, &imageCount, nullptr);

	context.swapChainImages.resize(imageCount);

	vkGetSwapchainImagesKHR(context.device, context.swapChain, &imageCount, context.swapChainImages.data());

	context.swapChainImageFormat = surfaceFormat.format;
	context.swapChainExtent = extent;
}

void DrawParticlesApp::createImageViews()
{
	context.swapChainImageViews.resize(context.swapChainImages.size());

	for (uint32_t i = 0; i < context.swapChainImages.size(); i++)
	{
		context.swapChainImageViews[i] = createImageView(context.swapChainImages[i], context.swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
	}
}

void DrawParticlesApp::createRenderPass()
{
	VkAttachmentDescription colorAttachmentDescription{};

	colorAttachmentDescription.format = context.swapChainImageFormat;
	colorAttachmentDescription.samples = context.msaaSamples;
	colorAttachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachmentDescription.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription depthAttachmentDescription{};

	depthAttachmentDescription.format = findDepthFormat();
	depthAttachmentDescription.samples = context.msaaSamples;
	depthAttachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	depthAttachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthAttachmentDescription.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription colorAttachmentResolveDescription{};

	colorAttachmentResolveDescription.format = context.swapChainImageFormat;
	colorAttachmentResolveDescription.samples = VK_SAMPLE_COUNT_1_BIT;
	colorAttachmentResolveDescription.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachmentResolveDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachmentResolveDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachmentResolveDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachmentResolveDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachmentResolveDescription.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference colorAttachmentReference{};

	colorAttachmentReference.attachment = 0;
	colorAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthAttachmentReference{};

	depthAttachmentReference.attachment = 1;
	depthAttachmentReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference colorAttachmentResolveReference{};

	colorAttachmentResolveReference.attachment = 2;
	colorAttachmentResolveReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpassDescription{};

	subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpassDescription.colorAttachmentCount = 1;
	subpassDescription.pResolveAttachments = &colorAttachmentResolveReference;
	subpassDescription.pDepthStencilAttachment = &depthAttachmentReference;
	subpassDescription.pColorAttachments = &colorAttachmentReference;

	VkSubpassDependency subpassDependency{};

	subpassDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	subpassDependency.dstSubpass = 0;
	subpassDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	subpassDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	subpassDependency.srcAccessMask = 0;
	subpassDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

	std::array<VkAttachmentDescription, 3> attachments = { colorAttachmentDescription, depthAttachmentDescription, colorAttachmentResolveDescription };

	VkRenderPassCreateInfo renderPassCreateInfo{};

	renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
	renderPassCreateInfo.pAttachments = attachments.data();
	renderPassCreateInfo.subpassCount = 1;
	renderPassCreateInfo.pSubpasses = &subpassDescription;
	renderPassCreateInfo.dependencyCount = 1;
	renderPassCreateInfo.pDependencies = &subpassDependency;

	if (vkCreateRenderPass(context.device, &renderPassCreateInfo, nullptr, &context.renderPass) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create render pass!");
	}
}

void DrawParticlesApp::createGraphicsPipeline()
{
	std::vector<char> vertShaderCode = readFile(vertShaderPath);
	std::vector<char> fragShaderCode = readFile(fragShaderPath);

	VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
	VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

	VkPipelineShaderStageCreateInfo vertShaderStageInfo{};

	vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertShaderStageInfo.module = vertShaderModule;
	vertShaderStageInfo.pName = "main";
	vertShaderStageInfo.pSpecializationInfo = nullptr;

	VkPipelineShaderStageCreateInfo fragShaderStageInfo{};

	fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragShaderStageInfo.module = fragShaderModule;
	fragShaderStageInfo.pName = "main";
	fragShaderStageInfo.pSpecializationInfo = nullptr;

	VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

	VkVertexInputBindingDescription bindingDescription = Particle::getBindingDescription();
	std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = Particle::getAttributeDescriptions();

	VkPipelineVertexInputStateCreateInfo vertexInputStateInfo{};

	vertexInputStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputStateInfo.vertexBindingDescriptionCount = 1;
	vertexInputStateInfo.pVertexBindingDescriptions = &bindingDescription;
	vertexInputStateInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
	vertexInputStateInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

	VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateInfo{};

	inputAssemblyStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssemblyStateInfo.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
	inputAssemblyStateInfo.primitiveRestartEnable = VK_FALSE;

	VkPipelineViewportStateCreateInfo viewportStateInfo{};

	viewportStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportStateInfo.viewportCount = 1;
	viewportStateInfo.scissorCount = 1;

	VkPipelineRasterizationStateCreateInfo rasterizationStateInfo{};

	rasterizationStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizationStateInfo.rasterizerDiscardEnable = VK_FALSE;
	rasterizationStateInfo.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizationStateInfo.lineWidth = 1.0f;
	rasterizationStateInfo.cullMode = VK_CULL_MODE_NONE;
	rasterizationStateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizationStateInfo.depthClampEnable = VK_FALSE;
	rasterizationStateInfo.depthBiasEnable = VK_FALSE;
	rasterizationStateInfo.depthBiasConstantFactor = 0.0f;
	rasterizationStateInfo.depthBiasClamp = 0.0f;
	rasterizationStateInfo.depthBiasSlopeFactor = 0.0f;

	VkPipelineMultisampleStateCreateInfo multisampleStateInfo{};

	multisampleStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampleStateInfo.rasterizationSamples = context.msaaSamples;
	multisampleStateInfo.sampleShadingEnable = VK_FALSE; // To enable/disable sample shading in the pipeline.
	multisampleStateInfo.minSampleShading = 1.0f; // Min fraction for sample shading; closer to one is smoother.
	multisampleStateInfo.pSampleMask = nullptr;
	multisampleStateInfo.alphaToCoverageEnable = VK_FALSE;
	multisampleStateInfo.alphaToOneEnable = VK_FALSE;

	VkPipelineDepthStencilStateCreateInfo depthStencilStateInfo{};

	depthStencilStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencilStateInfo.depthTestEnable = VK_FALSE;
	depthStencilStateInfo.depthWriteEnable = VK_TRUE;
	depthStencilStateInfo.depthCompareOp = VK_COMPARE_OP_ALWAYS;
	depthStencilStateInfo.depthBoundsTestEnable = VK_FALSE;
	depthStencilStateInfo.minDepthBounds = 0.0f;
	depthStencilStateInfo.maxDepthBounds = 1.0f;
	depthStencilStateInfo.stencilTestEnable = VK_FALSE;
	depthStencilStateInfo.front = {};
	depthStencilStateInfo.back = {};

	VkPipelineColorBlendAttachmentState colorBlendAttachment{};

	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_FALSE;
	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
	colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

	VkPipelineColorBlendStateCreateInfo colorBlendStateInfo{};

	colorBlendStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlendStateInfo.logicOpEnable = VK_FALSE;
	colorBlendStateInfo.logicOp = VK_LOGIC_OP_COPY;
	colorBlendStateInfo.attachmentCount = 1;
	colorBlendStateInfo.pAttachments = &colorBlendAttachment;
	colorBlendStateInfo.blendConstants[0] = 0.0f;
	colorBlendStateInfo.blendConstants[1] = 0.0f;
	colorBlendStateInfo.blendConstants[2] = 0.0f;
	colorBlendStateInfo.blendConstants[3] = 0.0f;

	std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

	VkPipelineDynamicStateCreateInfo dynamicStateInfo{};

	dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicStateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
	dynamicStateInfo.pDynamicStates = dynamicStates.data();

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};

	pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutCreateInfo.setLayoutCount = 1;
	pipelineLayoutCreateInfo.pSetLayouts = &context.descriptorSetLayout;
	pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
	pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;

	if (vkCreatePipelineLayout(context.device, &pipelineLayoutCreateInfo, nullptr, &context.graphicsPipelineLayout) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create graphics pipeline layout!");
	}

	VkGraphicsPipelineCreateInfo pipelineCreateInfo{};

	pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineCreateInfo.stageCount = 2;
	pipelineCreateInfo.pStages = shaderStages;
	pipelineCreateInfo.pVertexInputState = &vertexInputStateInfo;
	pipelineCreateInfo.pInputAssemblyState = &inputAssemblyStateInfo;
	pipelineCreateInfo.pViewportState = &viewportStateInfo;
	pipelineCreateInfo.pRasterizationState = &rasterizationStateInfo;
	pipelineCreateInfo.pMultisampleState = &multisampleStateInfo;
	pipelineCreateInfo.pDepthStencilState = &depthStencilStateInfo;
	pipelineCreateInfo.pColorBlendState = &colorBlendStateInfo;
	pipelineCreateInfo.pDynamicState = &dynamicStateInfo;
	pipelineCreateInfo.layout = context.graphicsPipelineLayout;
	pipelineCreateInfo.renderPass = context.renderPass;
	pipelineCreateInfo.subpass = 0;
	pipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
	pipelineCreateInfo.basePipelineIndex = -1;

	if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &context.graphicsPipeline) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create graphics pipeline!");
	}

	vkDestroyShaderModule(context.device, fragShaderModule, nullptr);
	vkDestroyShaderModule(context.device, vertShaderModule, nullptr);
}

void DrawParticlesApp::createComputePipeline()
{
	std::vector<char> compShaderCode = readFile(compShaderPath);
;
	VkShaderModule compShaderModule = createShaderModule(compShaderCode);

	VkPipelineShaderStageCreateInfo compShaderStageInfo{};

	compShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	compShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	compShaderStageInfo.module = compShaderModule;
	compShaderStageInfo.pName = "main";
	compShaderStageInfo.pSpecializationInfo = nullptr;

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};

	pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutCreateInfo.setLayoutCount = 1;
	pipelineLayoutCreateInfo.pSetLayouts = &context.descriptorSetLayout;
	pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
	pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;

	if (vkCreatePipelineLayout(context.device, &pipelineLayoutCreateInfo, nullptr, &context.computePipelineLayout) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create compute pipeline layout!");
	}

	VkComputePipelineCreateInfo pipelineCreateInfo{};

	pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineCreateInfo.layout = context.computePipelineLayout;
	pipelineCreateInfo.stage = compShaderStageInfo;

	if (vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &context.computePipeline) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create compute pipeline!");
	}

	vkDestroyShaderModule(context.device, compShaderModule, nullptr);
}

void DrawParticlesApp::createColorResources()
{
	VkFormat colorFormat = context.swapChainImageFormat;

	createImage(context.swapChainExtent.width, context.swapChainExtent.height, 1, context.msaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, context.colorImage, context.colorImageMemory);

	context.colorImageView = createImageView(context.colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
}

void DrawParticlesApp::createDepthResources()
{
	VkFormat depthFormat = findDepthFormat();

	createImage(context.swapChainExtent.width, context.swapChainExtent.height, 1, context.msaaSamples, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, context.depthImage, context.depthImageMemory);

	context.depthImageView = createImageView(context.depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
}

void DrawParticlesApp::createFramebuffers()
{
	context.swapChainFramebuffers.resize(context.swapChainImageViews.size());

	for (uint32_t i = 0; i < context.swapChainImageViews.size(); i++)
	{
		std::array<VkImageView, 3> attachments = { context.colorImageView, context.depthImageView, context.swapChainImageViews[i] };

		VkFramebufferCreateInfo framebufferCreateInfo{};

		framebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferCreateInfo.renderPass = context.renderPass;
		framebufferCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		framebufferCreateInfo.pAttachments = attachments.data();
		framebufferCreateInfo.width = context.swapChainExtent.width;
		framebufferCreateInfo.height = context.swapChainExtent.height;
		framebufferCreateInfo.layers = 1;

		if (vkCreateFramebuffer(context.device, &framebufferCreateInfo, nullptr, &context.swapChainFramebuffers[i]) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create framebuffer!");
		}
	}
}

void DrawParticlesApp::createCommandPool()
{
	QueueFamilyIndices queueFamilyIndices = findQueueFamilies(context.gpu);

	VkCommandPoolCreateInfo commandPoolCreateInfo{};

	commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndices.graphicsAndComputeFamily.value();

	if (vkCreateCommandPool(context.device, &commandPoolCreateInfo, nullptr, &context.commandPool) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create command pool!");
	}
}

void DrawParticlesApp::createGraphicsCommandBuffers()
{
	VkCommandBufferAllocateInfo commandBufferAllocateInfo{};

	context.graphicsCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

	commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	commandBufferAllocateInfo.commandPool = context.commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

	if (vkAllocateCommandBuffers(context.device, &commandBufferAllocateInfo, context.graphicsCommandBuffers.data()) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate graphics command buffers!");
	}
}

void DrawParticlesApp::createComputeCommandBuffers()
{
	VkCommandBufferAllocateInfo commandBufferAllocateInfo{};

	context.computeCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

	commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	commandBufferAllocateInfo.commandPool = context.commandPool;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

	if (vkAllocateCommandBuffers(context.device, &commandBufferAllocateInfo, context.computeCommandBuffers.data()) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate compute command buffers!");
	}
}

void DrawParticlesApp::createSyncObjects()
{
	VkSemaphoreCreateInfo semaphoreCreateInfo{};

	semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fenceCreateInfo{};

	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	context.swapChainAcquireSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	context.swapChainReleaseSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	context.graphicsSubmitFences.resize(MAX_FRAMES_IN_FLIGHT);

	context.computeFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	context.computeSubmitFences.resize(MAX_FRAMES_IN_FLIGHT);

	for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		bool error = false;

		error = error || vkCreateSemaphore(context.device, &semaphoreCreateInfo, nullptr, &context.swapChainAcquireSemaphores[i]) != VK_SUCCESS;
		error = error || vkCreateSemaphore(context.device, &semaphoreCreateInfo, nullptr, &context.swapChainReleaseSemaphores[i]) != VK_SUCCESS;
		error = error || vkCreateFence(context.device, &fenceCreateInfo, nullptr, &context.graphicsSubmitFences[i]) != VK_SUCCESS;

		if (error)
		{
			throw std::runtime_error("Failed to create graphics synchronization objects for a frame!");
		}
	}

	for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		bool error = false;

		error = error || vkCreateSemaphore(context.device, &semaphoreCreateInfo, nullptr, &context.computeFinishedSemaphores[i]) != VK_SUCCESS;
		error = error || vkCreateFence(context.device, &fenceCreateInfo, nullptr, &context.computeSubmitFences[i]) != VK_SUCCESS;

		if (error)
		{
			throw std::runtime_error("Failed to create compute synchronization objects for a frame!");
		}
	}
}

void DrawParticlesApp::createShaderStorageBuffers()
{
	float width = static_cast<float>(context.swapChainExtent.width);
	float height = static_cast<float>(context.swapChainExtent.height);
	std::default_random_engine rndEngine(static_cast<unsigned>(time(nullptr)));
	std::uniform_real_distribution<float> rndDistribution(0.0f, 1.0f);

	std::vector<Particle> particles(particleCount);

	for (auto& particle : particles)
	{
		float r = 0.25f * std::sqrt(rndDistribution(rndEngine));
		float theta = rndDistribution(rndEngine) * 2.0f * 3.14159265358979323846f;
		float x = r * std::cos(theta) * height / width;
		float y = r * std::sin(theta);

		particle.position = glm::vec2(x, y);
		particle.velocity = glm::normalize(glm::vec2(x, y)) * 0.00025f;
		particle.color = glm::vec4(rndDistribution(rndEngine), rndDistribution(rndEngine), rndDistribution(rndEngine), 1.0f);
	}

	VkDeviceSize bufferSize = sizeof(Particle) * particleCount;
	void* data;  // Allocated memory address.

	VkBuffer stagingBuffer{};
	VkDeviceMemory stagingBufferMemory{};
	
	createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

	vkMapMemory(context.device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, particles.data(), static_cast<size_t>(bufferSize));
	vkUnmapMemory(context.device, stagingBufferMemory);

	context.shaderStorageBuffers.resize(MAX_FRAMES_IN_FLIGHT);
	context.shaderStorageBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, context.shaderStorageBuffers[i], context.shaderStorageBuffersMemory[i]);
		
		copyBuffer(stagingBuffer, context.shaderStorageBuffers[i], bufferSize);
	}

	vkDestroyBuffer(context.device, stagingBuffer, nullptr);
	vkFreeMemory(context.device, stagingBufferMemory, nullptr);
}

void DrawParticlesApp::createUniformBuffers()
{
	VkDeviceSize bufferSize = sizeof(UniformBufferObject);

	context.uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
	context.uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
	context.uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, context.uniformBuffers[i], context.uniformBuffersMemory[i]);

		vkMapMemory(context.device, context.uniformBuffersMemory[i], 0, bufferSize, 0, &context.uniformBuffersMapped[i]);
	}
}

void DrawParticlesApp::createDescriptorSetLayout()
{
	VkDescriptorSetLayoutBinding uboLayoutBinding{};

	uboLayoutBinding.binding = 0;
	uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	uboLayoutBinding.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutBinding ssboInLayoutBinding{};

	ssboInLayoutBinding.binding = 1;
	ssboInLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	ssboInLayoutBinding.descriptorCount = 1;
	ssboInLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	ssboInLayoutBinding.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutBinding ssboOutLayoutBinding{};

	ssboOutLayoutBinding.binding = 2;
	ssboOutLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	ssboOutLayoutBinding.descriptorCount = 1;
	ssboOutLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	ssboOutLayoutBinding.pImmutableSamplers = nullptr;

	std::array<VkDescriptorSetLayoutBinding, 3> bindings = { uboLayoutBinding, ssboInLayoutBinding, ssboOutLayoutBinding };

	VkDescriptorSetLayoutCreateInfo layoutCreateInfo{};

	layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutCreateInfo.bindingCount = static_cast<uint32_t>(bindings.size());
	layoutCreateInfo.pBindings = bindings.data();

	if (vkCreateDescriptorSetLayout(context.device, &layoutCreateInfo, nullptr, &context.descriptorSetLayout) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create descriptor set layout!");
	}
}

void DrawParticlesApp::createDescriptorPool()
{
	std::array<VkDescriptorPoolSize, 2> poolSizes{};

	poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
	poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 2;

	VkDescriptorPoolCreateInfo poolCreateInfo{};

	poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolCreateInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	poolCreateInfo.pPoolSizes = poolSizes.data();
	poolCreateInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

	if (vkCreateDescriptorPool(context.device, &poolCreateInfo, nullptr, &context.descriptorPool) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create descriptor pool!");
	}
}

void DrawParticlesApp::createDescriptorSets()
{
	std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, context.descriptorSetLayout);

	VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};

	descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	descriptorSetAllocateInfo.descriptorPool = context.descriptorPool;
	descriptorSetAllocateInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
	descriptorSetAllocateInfo.pSetLayouts = layouts.data();

	context.descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);

	if (vkAllocateDescriptorSets(context.device, &descriptorSetAllocateInfo, context.descriptorSets.data()) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate descriptor sets!");
	}

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		VkDescriptorBufferInfo uniformBufferInfo{};

		uniformBufferInfo.buffer = context.uniformBuffers[i];
		uniformBufferInfo.offset = 0;
		uniformBufferInfo.range = sizeof(UniformBufferObject);

		VkDescriptorBufferInfo storageBufferInfoLastFrame{};

		storageBufferInfoLastFrame.buffer = context.shaderStorageBuffers[(i - 1) % MAX_FRAMES_IN_FLIGHT];
		storageBufferInfoLastFrame.offset = 0;
		storageBufferInfoLastFrame.range = sizeof(Particle) * particleCount;

		VkDescriptorBufferInfo storageBufferInfoCurrentFrame{};

		storageBufferInfoCurrentFrame.buffer = context.shaderStorageBuffers[i];
		storageBufferInfoCurrentFrame.offset = 0;
		storageBufferInfoCurrentFrame.range = sizeof(Particle) * particleCount;

		std::array<VkWriteDescriptorSet, 3> descriptorWrites{};

		descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[0].dstSet = context.descriptorSets[i];
		descriptorWrites[0].dstBinding = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo = &uniformBufferInfo;

		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[1].dstSet = context.descriptorSets[i];
		descriptorWrites[1].dstBinding = 1;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pBufferInfo = &storageBufferInfoLastFrame;

		descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[2].dstSet = context.descriptorSets[i];
		descriptorWrites[2].dstBinding = 2;
		descriptorWrites[2].dstArrayElement = 0;
		descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[2].descriptorCount = 1;
		descriptorWrites[2].pBufferInfo = &storageBufferInfoCurrentFrame;

		vkUpdateDescriptorSets(context.device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
	}
}

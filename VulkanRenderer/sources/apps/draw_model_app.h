#pragma once

#include "../application.h"

#include <tol/tiny_obj_loader.h>

class DrawModelApp : public Application
{
public:
	DrawModelApp();
	
	void setup(GLFWwindow* window);
	void cleanUp();

	void update(float deltaTime);
	void render(GLFWwindow* window, float deltaTime);

	struct Context
	{
		VkInstance instance = VK_NULL_HANDLE;

		VkPhysicalDevice gpu = VK_NULL_HANDLE;

		VkDevice device = VK_NULL_HANDLE;

		VkQueue graphicsQueue = VK_NULL_HANDLE;
		VkQueue presentQueue = VK_NULL_HANDLE;

		VkSurfaceKHR surface = VK_NULL_HANDLE;

		VkSwapchainKHR swapChain = VK_NULL_HANDLE;

		std::vector<VkImage> swapChainImages;
		std::vector<VkImageView> swapChainImageViews;
		VkFormat swapChainImageFormat;
		VkExtent2D swapChainExtent;

		VkImage colorImage = VK_NULL_HANDLE;
		VkDeviceMemory colorImageMemory = VK_NULL_HANDLE;
		VkImageView colorImageView = VK_NULL_HANDLE;

		VkImage depthImage = VK_NULL_HANDLE;
		VkDeviceMemory depthImageMemory = VK_NULL_HANDLE;
		VkImageView depthImageView = VK_NULL_HANDLE;

		std::vector<VkFramebuffer> swapChainFramebuffers;

		VkRenderPass renderPass = VK_NULL_HANDLE;

		VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

		VkPipeline graphicsPipeline = VK_NULL_HANDLE;

		VkCommandPool commandPool = VK_NULL_HANDLE;

		std::vector<VkCommandBuffer> commandBuffers;

		std::vector<VkSemaphore> swapChainAcquireSemaphores;
		std::vector<VkSemaphore> swapChainReleaseSemaphores;
		std::vector<VkFence> queueSubmitFences;

		VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

		VkBuffer vertexBuffer = VK_NULL_HANDLE;
		VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
		VkBuffer indexBuffer = VK_NULL_HANDLE;
		VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;

		std::vector<VkBuffer> uniformBuffers;
		std::vector<VkDeviceMemory> uniformBuffersMemory;
		std::vector<void*> uniformBuffersMapped;

		VkImage textureImage = VK_NULL_HANDLE;
		VkDeviceMemory textureImageMemory = VK_NULL_HANDLE;
		VkImageView textureImageView = VK_NULL_HANDLE;
		VkSampler textureSampler = VK_NULL_HANDLE;

		VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
		VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
		std::vector<VkDescriptorSet> descriptorSets;

		VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

		std::vector<Vertex> vertices;
		std::vector<uint32_t> indices;

		uint32_t mipLevels;

		uint32_t currentFrame = 0;
	};

private:
	Context context;

	std::string vertShaderPath = "sources/shaders/draw_model_vs.spv";
	std::string fragShaderPath = "sources/shaders/draw_model_fs.spv";
	std::string texturePath = "resources/models/viking_room/viking_room.png";
	std::string modelPath = "resources/models/viking_room/viking_room.obj";

	void logExtensionSupport();
	bool checkValidationLayerSupport();
	std::vector<const char*> getRequiredInstanceExtensions();

	bool checkDeviceExtensionSupport(VkPhysicalDevice device);
	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
	bool isDeviceSuitable(VkPhysicalDevice device);
	VkSampleCountFlagBits getMaxUsableSampleCount();
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
	VkExtent2D chooseSwapExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities);
	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels);
	VkShaderModule createShaderModule(const std::vector<char>& code);
	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
	VkFormat findDepthFormat();
	bool hasStencilComponent(VkFormat format);

	void cleanUpSwapChain();
	void recreateSwapChain(GLFWwindow* window);

	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);

	VkCommandBuffer beginSingleTimeCommands();
	void endSingleTimeCommands(VkCommandBuffer commandBuffer);
	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
	void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels);
	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
	void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);

	void updateUniformBuffer(uint32_t currentImage);

	void loadModel();

	void createInstance();
	void createDebugMessenger();
	void createSurface(GLFWwindow* window);

	void selectPhysicalDevice();
	void createLogicalDevice();

	void createSwapChain(GLFWwindow* window);
	void createImageViews();

	void createRenderPass();
	void createGraphicsPipeline();

	void createColorResources();
	void createDepthResources();

	void createFramebuffers();

	void createCommandPool();
	void createCommandBuffers();

	void createSyncObjects();

	void createVertexBuffer();
	void createIndexBuffer();

	void createTextureImage();
	void createTextureImageView();
	void createTextureSampler();

	void createUniformBuffers();
	void createDescriptorSetLayout();
	void createDescriptorPool();
	void createDescriptorSets();
};

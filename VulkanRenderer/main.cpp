// Vulkan Renderer.

#define GLFW_INCLUDE_VULKAN
#define STB_IMAGE_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION

#include <GLFW/glfw3.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "sources/application.h"
#include "sources/apps/draw_model_app.h"
#include "sources/apps/draw_particles_app.h"

enum AppIdentifier
{
	DRAW_MODEL, DRAW_PARTICLES
};

class Program
{
public:
	void run(int appIdentifier)
	{
		setup(appIdentifier);
		runMainLoop();
		cleanUp();
	}

private:
	uint32_t windowWidth = 1600, windowHeight = 900;

	GLFWwindow* window;

	Application* app;

	float deltaTime = 0.0f, lastFrame = 0.0f;

	static void framebufferSizeCallback(GLFWwindow* window, int width, int height)
	{
		Application* ptr = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));

		ptr->framebufferResized = true;
	}

	void setup(int appIdentifier)
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(windowWidth, windowHeight, "Vulkan Renderer", nullptr, nullptr);

		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

		switch (appIdentifier)
		{
		case AppIdentifier::DRAW_MODEL:
			app = new DrawModelApp();
			break;

		case AppIdentifier::DRAW_PARTICLES:
			app = new DrawParticlesApp();
			break;

		default:
			break;
		}

		app->setup(window);
	}

	void runMainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			float currTime = static_cast<float>(glfwGetTime());

			deltaTime = currTime - lastFrame;
			lastFrame = currTime;

			glfwPollEvents();

			app->update(deltaTime);
			app->render(window, deltaTime);
		}
	}

	void cleanUp()
	{
		app->cleanUp();

		glfwDestroyWindow(window);

		glfwTerminate();
	}
};

int main()
{
	Program program;
	int appIdentifier = -1;

	try
	{
		std::cout << "APPLICATIONS:" << std::endl;
		std::cout << "\t0. DRAW MODEL" << std::endl;
		std::cout << "\t1. DRAW PARTICLES" << std::endl;
		std::cout << "> ";

		std::cin >> appIdentifier;

		program.run(appIdentifier);
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;

		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
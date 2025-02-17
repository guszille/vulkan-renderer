#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragmentColor;
layout(location = 1) out vec2 fragmentTexCoord;

layout(binding = 0) uniform UniformBufferObject
{
    mat4 model;
    mat4 view;
    mat4 projection;
} UBO;

void main()
{
    gl_Position = UBO.projection * UBO.view * UBO.model * vec4(inPosition, 1.0);

    fragmentColor = inColor;
    fragmentTexCoord = inTexCoord;
}

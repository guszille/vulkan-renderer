#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec3 fragmentColor;

void main()
{
    gl_Position = vec4(inPosition.xy, 1.0, 1.0);
    gl_PointSize = 14.0;

    fragmentColor = inColor.rgb;
}

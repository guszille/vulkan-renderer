#version 450

layout(location = 0) in vec3 fragmentColor;

layout(location = 0) out vec4 outColor;

void main()
{
    vec2 coord = gl_PointCoord - vec2(0.5);

    outColor = vec4(fragmentColor, 0.5 - length(coord));
}

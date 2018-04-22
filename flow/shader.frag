#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 color;

layout(location = 0) out vec4 o_color;

layout(push_constant) uniform PushConstants 
{
	float time;
	float padding;
	vec2 resolution;
} push_constants;

void main() 
{
	float p = push_constants.padding;
	
	float modulate = sin(push_constants.time) * 0.5 + 0.5;
	vec2 screen_color = gl_FragCoord.xy / push_constants.resolution;

    o_color = vec4(screen_color * modulate, 0.0, 1.0);
}
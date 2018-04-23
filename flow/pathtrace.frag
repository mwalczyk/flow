#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform sampler2D u_image;

layout(push_constant) uniform PushConstants 
{
	float time;
	float frame_counter;
	vec2 resolution;
} push_constants;

layout(location = 0) out vec4 o_color;

void main() 
{	
	vec2 uv = gl_FragCoord.xy / push_constants.resolution;

	float radius = 0.1;
	float smoothing = 0.01;
	vec2 center = vec2(sin(push_constants.time), cos(push_constants.time)) * 0.25 + vec2(0.5);
	float pct = 1.0 - smoothstep(radius, radius + radius * smoothing, distance(uv, center));

	vec4 prev_frame = texture(u_image, uv);
	vec4 curr_frame = vec4(vec3(pct), 1.0);
	o_color = prev_frame + curr_frame;
}
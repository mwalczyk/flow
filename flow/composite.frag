#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (input_attachment_index = 0, set = 0, binding = 1) uniform subpassInput u_rendered;

layout(push_constant) uniform PushConstants 
{
	float time;
	float frame_counter;
	vec2 resolution;
	vec2 cursor_position;
} push_constants;

layout(location = 0) out vec4 o_color;

void main() 
{
	vec2 uv = gl_FragCoord.xy / push_constants.resolution;

	vec3 total = subpassLoad(u_rendered).rgb;

	// Normalize
	// if (push_constants.frame_counter > 0.0)
	// {
	// 	total /= push_constants.frame_counter + 1.0;
	// }

	//if (push_constants.frame_counter > 2.0)
	{
		total /= push_constants.frame_counter + 4.0;
	}
	
    o_color = vec4(total, 1.0);
}
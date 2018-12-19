#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable 
#extension GL_GOOGLE_include_directive : require

#include "utilities.glsl"
#include "primitives.glsl"
#include "intersections.glsl"
#include "materials.glsl"

layout(set = 0, binding = 0) uniform sampler2D u_image;

layout(push_constant) uniform PushConstants 
{
	float time;
	float frame_counter;
	vec2 resolution;
	vec2 cursor_position;
	float mouse_down;
} push_constants;

layout(location = 0) out vec4 o_color;

/****************************************************************************************************
 *
 * Scene Definition
 *
 ***************************************************************************************************/
material materials[] = 
{
	{ { 0.90f, 0.80f, 0.10f }, material_type_metallic, false }, 
	{ { 0.90f, 0.10f, 0.20f }, material_type_diffuse, false },
	{ { 0.97f, 1.00f, 0.97f }, material_type_diffuse, false }, 	// Off-white

	{ { 1.00f, 0.35f, 0.37f }, material_type_diffuse, false }, 	// Pink
    { { 0.54f, 0.79f, 0.15f }, material_type_diffuse, false }, 	// Mint
    { { 0.10f, 0.51f, 0.77f }, material_type_diffuse, false }, 	// Dark mint
	{ { 1.00f, 0.79f, 0.23f }, material_type_diffuse, false },	// Yellow
	{ { 0.42f, 0.30f, 0.58f }, material_type_diffuse, false }, 	// Purple

	{ { 5.00f, 5.00f, 5.00f }, material_type_diffuse, true },  	// Light
};

sphere spheres[] = 
{
	{ 0.55, vec3( -1.4,  0.4, -1.3), 0 },
	{ 0.20, vec3(  1.0,  0.7, -1.6), 1 },
	{ 1.50, vec3(  0.0, -0.6,  0.0), 2 },
	{ 0.50, vec3( -1.4, -2.6, -1.5), 8 }
};

plane planes[] = 
{
	{ -z_axis,  z_axis * 5.5, 6 }, // Back
	{  z_axis, -z_axis * 8.5, 2 }, // Front
	{ -x_axis,  x_axis * 4.5, 3 }, // Left
	{  x_axis, -x_axis * 4.5, 4 }, // Right
	{  y_axis, -y_axis * 4.5, 7 }, // Top
	{ -y_axis,  y_axis * 1.0, 2 }  // Bottom
};

/****************************************************************************************************
 *
 * Tracing Routine
 *
 ***************************************************************************************************/
intersection intersect_scene(in ray r)
{
	// This function simply iterates through all of the objects in our scene, 
	// keeping track of the closest intersection (if any). For shading and
	// subsequent ray bounces, we need to keep track of a bunch of information
	// about the object that was hit, like it's normal vector at the point
	// of intersection.
	//
	// Initially, we assume that we won't hit anything by setting the fields
	// of our `intersection` struct as follows:
	intersection inter = 
	{
		-1,
		object_type_miss,
		-1,
		r.direction,
		r.origin,
		{ -1.0f, -1.0f, -1.0f },
		max_distance
	};

	// Now, let's iterate through the objects in our scene, starting with 
	// the spheres:
	for(uint i = 0u; i < spheres.length(); ++i)
	{	
		// Did the ray intersect this object?
		float temp_t;
		if (intersect_sphere(spheres[i], r, temp_t))
		{
			// Was the intersection closer than any previous one?
			if (temp_t < inter.t)
			{
				inter.material_index = spheres[i].material_index;
				inter.object_type = object_type_sphere;
				inter.object_index = int(i);
				inter.position = r.origin + r.direction * temp_t;
				inter.normal = normalize(inter.position - spheres[i].center);
				inter.t = temp_t;
			}
		} 
	}

	// then planes:
	for(uint i = 0u; i < planes.length(); ++i)
	{	
		// Did the ray intersect this object?
		float temp_t;
		if (intersect_plane(planes[i], r, temp_t))
		{
			// Was the intersection closer than any previous one?
			if (temp_t < inter.t)
			{
				inter.material_index = planes[i].material_index;
				inter.object_type = object_type_plane;
				inter.object_index = int(i);
				inter.position = r.origin + r.direction * temp_t;
				inter.normal = normalize(planes[i].normal);
				inter.t = temp_t;
			}
		} 
	}

	// TODO quads
	// ...

	return inter;
}

vec3 scatter(in material mtl, in intersection inter, float rand_a, float rand_b)
{
	// Return the new ray direction

	return vec3(0.0f);
}

ray get_ray(in vec2 uv)
{
	vec3 o = vec3(1.0f);
	vec3 d = vec3(1.0f);
	return ray(o, d);
}

#define  RUSSIAN_ROULETTE

vec3 trace()
{
	// We want to make sure that we correct for the window's aspect ratio
	// so that our final render doesn't look skewed or stretched when the
	// resolution changes.
	const float aspect_ratio = push_constants.resolution.x / push_constants.resolution.y;
	vec2 uv = gl_FragCoord.xy / push_constants.resolution;

	float t = push_constants.time;
	vec4 seed = { uv.x + t, 
	              uv.y + t, 
	              uv.x - t, 
	              uv.y - t };
	
	vec3 final = black;
	
	vec3 offset = vec3(0.0f);// vec3(push_constants.cursor_position * 2.0f - 1.0f, 0.0f) * 8.0f;
	vec3 camera_position = vec3(0.0, -3.0, -10.5) + offset;

	const float vertical_fov = 60.0f;
	const float aperture = 0.5f;
	const float lens_radius = aperture / 2.0f;
	const float theta = vertical_fov * pi / 180.0f;
	const float half_height = tan(theta * 0.5f);
	const float half_width = aspect_ratio * half_height;

	const mat3 look_at = lookat(camera_position, origin);
	const float dist_to_focus = length(camera_position);

	const vec3 lower_left_corner = camera_position - look_at * vec3(half_width, half_height, 1.0f) * dist_to_focus;
	const vec3 horizontal = 2.0f * half_width * dist_to_focus * look_at[0];
	const vec3 vertical = 2.0f * half_height * dist_to_focus * look_at[1];

	for (uint j = 0; j < number_of_iterations; ++j)
	{
		// By jittering the uv-coordinates a tiny bit here, we get "free" anti-aliasing.
		vec2 jitter = { gpu_rnd(seed), gpu_rnd(seed) };
		jitter = jitter * 2.0 - 1.0;
		uv += (jitter / push_constants.resolution) * anti_aliasing;
		
		// Depth-of-field calculation.
		vec3 rd = lens_radius * vec3(random_on_disk(seed), 0.0f);
		vec3 lens_offset = look_at * vec3(rd.xy, 0.0f);
		vec3 ro = camera_position + lens_offset;
		rd = lower_left_corner + uv.x * horizontal + uv.y * vertical - camera_position - lens_offset;

		// Calculate the ray direction based on the current fragment's uv-coordinates. All 
		// rays will originate from the camera's location. 		
		ray r = { ro, rd };

		// Define some colors.
		const vec3 sky = black;
		vec3 radiance = black;
		vec3 throughput = white;

		// For explicit light sampling (next event estimation), we need to keep track of the 
		// previous material that was hit, as explained below.
		int prev_material_type = 0;

		// This is the main path tracing loop.
		for (uint i = 0; i < number_of_bounces; ++i)
		{	
			intersection inter = intersect_scene(r);

            // There were no intersections: simply accumulate the background color and break.
            if (inter.object_type == object_type_miss) 
            {
                radiance += throughput * sky;   
                break;
            }
            // There was an intersection: accumulate color and bounce.
            else 
            {
            	material mtl = materials[inter.material_index];

                const vec3 hit_location = r.origin + r.direction * inter.t;

                // When using explicit light sampling, we have to account for a number of edge cases:
                //
                // 1. If this is the first bounce, and the object we hit is a light, we need to add its
                //    color (otherwise, lights would appear black in the final render).
                // 2. If the object we hit is a light, and the PREVIOUS object we hit was specular (a
                //    metal or dielectric), we need to add its color (otherwise, lights would appear
                //    black in mirror-like objects).
                if ((j == 0 || 
                    prev_material_type == material_type_dielectric || 
                    prev_material_type == material_type_metallic) && 
                    mtl.is_light) 
                {
                    radiance += throughput * mtl.reflectance;   
                }
            
                // Set the new ray origin.
                r.origin = hit_location + inter.normal * epsilon;

                // Choose a new ray direction based on the material type.
                if (mtl.type == material_type_diffuse)
                {

                    // Sample all of the light sources (right now we only have one)
                    // ...

                    float cos_a_max = 0.0f;
                    const vec3 light_position = spheres[3].center;
                    const float light_radius = spheres[3].radius;
                    vec3 to_light = sample_sphere_light(seed, cos_a_max, light_position, light_radius, r.origin);

                    // Items resulting from the shadow ray.
                    const ray secondary_r = { r.origin, to_light };
                    const intersection secondary_inter = intersect_scene(secondary_r);
                    const material secondary_mtl = materials[secondary_inter.material_index];

                    if (secondary_inter.t > 0.0f &&      						// We hit an object
                        secondary_mtl.is_light && 								// ...and that object was the light source 
                        !mtl.is_light)      									// ...and the original object wasn't the light source (avoid self-intersection).  
                    {
                        const float omega = (1.0f - cos_a_max) * two_pi;
                        const vec3 normal_towards_light = dot(inter.normal, r.direction) < 0.0f ? inter.normal : -inter.normal;

                        const float pdf = max(0.0f, dot(to_light, normal_towards_light)) * omega * one_over_pi;

                        radiance += throughput * secondary_mtl.reflectance * pdf; 
                    }

                    r.direction = normalize(cos_weighted_hemisphere(inter.normal, gpu_rnd(seed), gpu_rnd(seed)));

                    // Accumulate material color
                    const float cos_theta = max(0.0f, dot(inter.normal, r.direction));
                    const float pdf = one_over_two_pi;
                    const vec3 brdf = mtl.reflectance * one_over_pi;

                    throughput *= (brdf / pdf) * cos_theta;  
                }
                else if (mtl.type == material_type_metallic)
                {
                    const float roughness = 0.0f;
                    const vec3 offset = cos_weighted_hemisphere(inter.normal, gpu_rnd(seed), gpu_rnd(seed)) * roughness;
                    r.direction = normalize(reflect(r.direction, inter.normal) + offset);
                    
                    throughput *= mtl.reflectance;      
                }
                else if (mtl.type == material_type_dielectric)
                {
                    // Snell's Law states:
                    //
                    //      n * sin(Ѳ) = n' * sin(Ѳ')
                    //
                    // where n is the "outer" medium and n' is the "inner" (i.e. the medium
                    // that the ray is traveling into)
                    const float ior = 1.0f / 1.31f;

                    const vec3 normal = inter.normal;

                    vec3 outward_normal;

                    float ni_over_nt;
                    float cosine;

                    if (dot(normal, r.direction) > 0.0f) 
                    {
                        // We are inside the medium: flip the outward-facing normal
                        outward_normal = -normal;
                        ni_over_nt = 1.0f / ior;
                        cosine = ni_over_nt * dot(r.direction, normal) / length(r.direction);
                    } 
                    else 
                    {
                        outward_normal = normal;
                        ni_over_nt = ior;
                        cosine = -dot(r.direction, normal) / length(r.direction);
                    }

                    const vec3 reflected = reflect(r.direction, normal);
                    const vec3 refracted = refract(r.direction, outward_normal, ni_over_nt);

                    // Check for total internal reflection
                    float probability_of_reflection = (refracted == vec3(0.0f)) ? 1.0f : schlick(cosine, ior);

                    // Set new ray origin and direction
                    r.origin = hit_location + r.direction * epsilon;
                    r.direction = (gpu_rnd(seed) < probability_of_reflection) ? reflected : refracted;

                    // TODO: Fresnel effect
                    //throughput *= payload_pri.albedo;
                }

                prev_material_type = mtl.type;
#ifdef RUSSIAN_ROULETTE
                // See: https://computergraphics.stackexchange.com/questions/2316/is-russian-roulette-really-the-answer
                //
                // Bright objects (higher throughput) encourage more bounces
                const float probality_of_termination = max3(throughput);

                if (gpu_rnd(seed) > probality_of_termination) break;

                // Make sure the final render is unbiased
                throughput *= 1.0f / probality_of_termination;
#endif
            }
		}
		final += radiance; 
	}
	return final / float(number_of_iterations);
}

void main()
{	
	vec3 trace_color = trace();

	// Perform gamma correction.
	trace_color = pow(trace_color, vec3(gamma));

	vec2 uv = gl_FragCoord.xy / push_constants.resolution;
	vec3 prev_frame = texture(u_image, uv).rgb;
	vec3 curr_frame = trace_color;

	if (push_constants.mouse_down == 1.0f)
	{
		o_color = vec4(curr_frame, 1.0f);
	}
	else
	{
		o_color = vec4(prev_frame + curr_frame, 1.0f);
	}
}
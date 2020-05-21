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
	vec4 cursor_position;
	vec2 mouse_down;
} push_constants;

layout(location = 0) out vec4 o_color;

// Define the scene.
material materials[] = 
{
	{ { 0.9f, 0.9f, 0.9f }, material_type_metallic, 0.50f, false }, 
	{ { 0.90f, 0.10f, 0.20f }, material_type_dielectric, 1.0f / 2.31f, false },
	{ { 0.97f, 1.00f, 0.97f }, material_type_diffuse, _ignored, false }, 		// Off-white

	{ { 1.00f, 0.00f, 0.00f }, material_type_diffuse, _ignored, false }, 		// Pink
    { { 0.54f, 0.79f, 0.15f }, material_type_diffuse, _ignored, false }, 		// Mint
    { { 0.10f, 0.51f, 0.77f }, material_type_diffuse, _ignored, false }, 		// Dark mint
	{ { 1.00f, 0.79f, 0.23f }, material_type_diffuse, _ignored, false },		// Yellow
	{ { 0.42f, 0.30f, 0.58f }, material_type_diffuse, _ignored, false }, 		// Purple

	{ { 5.00f, 4.80f, 4.80f }, material_type_diffuse, _ignored, true },  		// *Light

	{ { 1.00f, 0.98f, 0.98f }, material_type_metallic, 0.25f, false }, 
	{ { 1.00f, 0.98f, 0.98f }, material_type_metallic, 0.50f, false }, 
	{ { 1.00f, 0.98f, 0.98f }, material_type_metallic, 0.75f, false }, 
	{ { 1.00f, 0.98f, 0.98f }, material_type_metallic, 1.00f, false }, 
	{ { 1.00f, 0.98f, 0.98f }, material_type_metallic, 1.25f, false }, 
	{ { 1.00f, 0.98f, 0.98f }, material_type_metallic, 1.50f, false }, 
	{ { 1.00f, 0.98f, 0.98f }, material_type_metallic, 1.75f, false }, 

	{ { 0.90f, 0.10f, 0.20f }, material_type_dielectric, 1.0f / 1.31f, false },

	{ { 0.3f, 0.3f, 0.3f }, material_type_metallic, 0.50f, false },
};

sphere spheres[] = 
{
	// Light source 
	{ 0.75f, vec3( -2.00f, -4.00f,  0.00f), 8 },

	// Spheres in the middle
	{ 1.500f, vec3( 0.00f, -1.500f,  0.000f), 1 },
	{ 0.750f, vec3( 0.00f, -0.750f, -2.250f), 4 },
    { 0.375f, vec3( 0.00f, -0.375f, -3.375f), 5 },
	{ 0.100f, vec3( 0.00f, -0.100f, -3.850f), 6 },

	// // Line of spheres on the left
	{ 0.30f, vec3(-3.00f, -0.30f, -2.00f),  9 },
	{ 0.30f, vec3(-3.00f, -0.30f, -1.00f), 10 },
	{ 0.30f, vec3(-3.00f, -0.30f,  0.00f), 11 },
	{ 0.30f, vec3(-3.00f, -0.30f,  1.00f), 12 },
	{ 0.30f, vec3(-3.00f, -0.30f,  2.00f), 13 },
	{ 0.30f, vec3(-3.00f, -0.30f,  3.00f), 14 },
	{ 0.30f, vec3(-3.00f, -0.30f,  4.00f), 15 }
};

plane planes[] = 
{
	// Colored
	{ -z_axis,  z_axis * 5.50f, 5 }, // Back
	{  z_axis, -z_axis * 8.50f, 2 }, // Front
	{ -x_axis,  x_axis * 4.50f, 7 }, // Right
	{  x_axis, -x_axis * 4.50f, 4 }, // Left
	{  y_axis, -y_axis * 7.00f, 2 }, // Top
	{ -y_axis,  y_axis * 0.00f, 2 }  // Bottom
};

// These intersection routines are quite slow, so let's disable them for now:
//
// #define QUADS
// #define BOXES
//
// quad quads[] =
// {
// 	build_quad(7.00f, 1.00f, vec3(3.00f, -0.60f, 0.00f), y_axis, to_radians(90.0f),  9),
// 	build_quad(7.00f, 1.00f, vec3(3.00f, -1.70f, 0.00f), y_axis, to_radians(90.0f), 12),
// 	build_quad(7.00f, 1.00f, vec3(3.00f, -2.80f, 0.00f), y_axis, to_radians(90.0f), 15),
// };

// box boxes[] =
// {
// 	build_box(vec3(2.0f, 4.0f, 2.0f), -y_axis * 2.0f, 3)
// };

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

	// // then quads:
#ifdef QUADS
	for(uint i = 0u; i < quads.length(); ++i)
	{	
		// Did the ray intersect this object?
		float temp_t;
		if (intersect_quad(quads[i], r, temp_t))
		{
			// Was the intersection closer than any previous one?
			if (temp_t < inter.t)
			{
				const vec3 edge0 = quads[i].ur - quads[i].ul;
				const vec3 edge1 = quads[i].ll - quads[i].ul;
				const vec3 normal = normalize(cross(edge0, edge1));

				inter.material_index = quads[i].material_index;
				inter.object_type = object_type_quad;
				inter.object_index = int(i);
				inter.position = r.origin + r.direction * temp_t;
				inter.normal = normal;
				inter.t = temp_t;
			}
		} 
	}
#endif

	// finally, boxes:
#ifdef BOXES
	for(uint i = 0u; i < boxes.length(); ++i)
	{	
		// Did the ray intersect this object?
		float temp_t;
		if (intersect_box(boxes[i], r, temp_t))
		{
			// Was the intersection closer than any previous one?
			if (temp_t < inter.t)
			{
				inter.material_index = boxes[i].material_index;
				inter.object_type = object_type_box;
				inter.object_index = int(i);
				inter.position = r.origin + r.direction * temp_t;

				vec3 normal;
				{
					vec3 center = (boxes[i].bounds[0] + boxes[i].bounds[1]) * 0.5f; 
					vec3 size = abs(boxes[i].bounds[1] - boxes[i].bounds[0]);
					vec3 local_point = inter.position - center;
					float min = 10000.0f;

					// Find which face is the closest to the point of 
					// intersection (x, y, or z).
					float dist = abs(size.x - abs(local_point.x));
					if (dist < min)
					{
						min = dist;
						normal = x_axis;
						normal *= sign(local_point.x);
					}

					dist = abs(size.y - abs(local_point.y));
					if (dist < min)
					{
						min = dist;
						normal = y_axis;
						normal *= sign(local_point.y);
					}

					dist = abs(size.z - abs(local_point.z));
					if (dist < min)
					{
						min = dist;
						normal = z_axis;
						normal *= sign(local_point.z);
					}
				}
				
				inter.normal = normalize(normal);
				inter.t = temp_t;
			}
		} 
	}
#endif

	return inter;
}

vec3 scatter(in material mtl, in intersection inter)
{
	// TODO: return the new ray direction
	// ...

	return vec3(0.0f);
}

ray get_ray(inout vec4 seed, in vec2 uv)
{
	// We want to make sure that we correct for the window's aspect ratio
	// so that our final render doesn't look skewed or stretched when the
	// resolution changes.
	const float aspect_ratio = push_constants.resolution.x / push_constants.resolution.y;

	const vec3 offset = vec3(push_constants.cursor_position.xy * 2.0f - 1.0f, 0.0f) * 8.0f;
	const vec3 camera_position = vec3(0.0f, -4.0f, -10.5f) + offset;

	const float vertical_fov = 60.0f;
	const float aperture = 0.5f;
	const float lens_radius = aperture * 0.5f;
	const float theta = to_radians(vertical_fov);
	const float half_height = tan(theta * 0.5f);
	const float half_width = aspect_ratio * half_height;

	const mat3 look_at = lookat(camera_position, origin + vec3(0.0f, -1.25f, 0.0f));
	const float dist_to_focus = map(push_constants.cursor_position.w, 0.0f, 1.0f, 10.0f, 5.0f);

	const vec3 lower_left_corner = camera_position - look_at * vec3(half_width, half_height, 1.0f) * dist_to_focus;
	const vec3 horizontal = 2.0f * half_width * dist_to_focus * look_at[0];
	const vec3 vertical = 2.0f * half_height * dist_to_focus * look_at[1];

	// By jittering the uv-coordinates a tiny bit here, we get "free" anti-aliasing.
	vec2 jitter = { gpu_rnd(seed), gpu_rnd(seed) };
	jitter = jitter * 2.0f - 1.0f;
	uv += (jitter / push_constants.resolution) * anti_aliasing;
	
	// Depth-of-field calculation.
	vec3 rd = lens_radius * vec3(random_on_disk(seed), 0.0f);
	vec3 lens_offset = look_at * vec3(rd.xy, 0.0f);
	vec3 ro = camera_position + lens_offset;
	rd = lower_left_corner + uv.x * horizontal + uv.y * vertical - camera_position - lens_offset;

	rd = normalize(rd);

	return ray(ro, rd);
}

#define RUSSIAN_ROULETTE

vec3 trace()
{
	vec2 uv = gl_FragCoord.xy / push_constants.resolution;

	float t = push_constants.time;
	vec4 seed = { uv.x + t, uv.y + t, uv.x - t, uv.y - t };
	
	vec3 final = black;
	const vec3 sky = black;
	const bool debug = false;

	for (uint i = 0; i < number_of_iterations; ++i)
	{
		// Calculate the ray direction based on the current fragment's uv-coordinates. All 
		// rays will originate from the camera's location. 		
		ray r = get_ray(seed, uv);

		// Define some colors.
		vec3 radiance = black;
		vec3 throughput = white;

		// For explicit light sampling (next event estimation), we need to keep track of the 
		// previous material that was hit, as explained below.
		int previous_material_type = 0;

		// This is the main path tracing loop.
		for (uint j = 0; j < number_of_bounces; ++j)
		{	
			intersection inter = intersect_scene(r);

            // There were no intersections: simply accumulate the background color and break.
            if (inter.object_type == object_type_miss) 
            {
                radiance += throughput * sky;   
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
                    previous_material_type == material_type_dielectric || 
                    previous_material_type == material_type_metallic) && 
                    mtl.is_light) 
                {
                	const vec3 light_color = white;

                    radiance += throughput * light_color;
                    break;   
                }

                if (debug)
                {
	                radiance += inter.normal * 0.5 + 0.5;
	                break;
         	    }
            
                // Set the new ray origin.
                r.origin = hit_location + inter.normal * epsilon;

                // Choose a new ray direction based on the material type. 
                if (mtl.type == material_type_diffuse)
                {

                    // Here, we explicitly sample each of the light sources in our scene.
                    // Obviously, for scenes with many lights, this would be prohibitively 
                    // expensive. So, if we're using `n` lights, we randomly choose one
                    // light to sample and divide its contribution by `n`.
                    // ...

                    float cos_a_max = 0.0f;
                    const vec3 to_light = sample_sphere_light(seed, cos_a_max, spheres[0].center, spheres[0].radius, r.origin);

                    // Items resulting from the shadow ray.
                    const ray secondary_r = { r.origin, to_light };
                    const intersection secondary_inter = intersect_scene(secondary_r);
                    const material secondary_mtl = materials[secondary_inter.material_index];

                    if (secondary_inter.t > 0.0f &&     // We hit an object
                        secondary_mtl.is_light && 		// ...and that object was the light source 
                        !mtl.is_light)      			// ...and the original object wasn't the light source (avoid self-intersection).  
                    {
                        const float omega = (1.0f - cos_a_max) * two_pi;
                        const vec3 normal_towards_light = dot(inter.normal, r.direction) < 0.0f ? inter.normal : -inter.normal;

                        const float pdf = max(0.0f, dot(to_light, normal_towards_light)) * omega * one_over_pi;

                        const vec3 light_color = white * 5.0f;

                        radiance += throughput * light_color * pdf; 
                    }

                    r.direction = normalize(cos_weighted_hemisphere(inter.normal, gpu_rnd(seed), gpu_rnd(seed)));

                    // Accumulate the material color.
                    const float cos_theta = max(0.0f, dot(inter.normal, r.direction));
                    const float pdf = one_over_two_pi;
                    const vec3 brdf = mtl.reflectance * one_over_pi;

                    throughput *= (brdf / pdf) * cos_theta;  
                }
                else if (mtl.type == material_type_metallic)
                {
                    const vec3 offset = cos_weighted_hemisphere(inter.normal, gpu_rnd(seed), gpu_rnd(seed)) * mtl.roughness;
                    
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
                    const float ior = mtl.roughness;

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

                    r.origin = hit_location + r.direction * epsilon;
                    r.direction = (gpu_rnd(seed) < probability_of_reflection) ? reflected : refracted;

                    r.direction = normalize(r.direction);

                    // TODO: Fresnel effect
                    // ...
                }

                previous_material_type = mtl.type;
#ifdef RUSSIAN_ROULETTE
                // See: https://computergraphics.stackexchange.com/questions/2316/is-russian-roulette-really-the-answer
                //
                // Bright objects (higher throughput) encourage more bounces.
                const float probality_of_termination = max3(throughput);

                if (gpu_rnd(seed) > probality_of_termination) break;

                // Make sure the final render is unbiased.
                throughput *= 1.0f / probality_of_termination;
#endif
            }
		}
#ifdef CLAMP_FIREFLIES
		radiance = clamp(vec3(0.0f), vec3(1.0f), radiance);
#endif
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

	if (any(bvec2(push_constants.mouse_down)))
	{
		o_color = vec4(curr_frame, 1.0f);
	}
	else
	{
		o_color = vec4(prev_frame + curr_frame, 1.0f);
	}
}
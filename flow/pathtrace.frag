#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable 

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
 * Scalar Constants
 *
 ***************************************************************************************************/
const float pi = 3.1415926535897932384626433832795f;
const float two_pi = pi * 2.0f;
const float gamma = 1.0f / 2.2f;
const float anti_aliasing = 0.5f;
const uint number_of_iterations = 6;
const uint number_of_bounces = 4;
const float epsilon = 0.001f;
const float max_distance = 1000.0f;

/****************************************************************************************************
 *
 * Vector Constants
 *
 ***************************************************************************************************/
const vec3 x_axis = { 1.0, 0.0, 0.0 };
const vec3 y_axis = { 0.0, 1.0, 0.0 };
const vec3 z_axis = { 0.0, 0.0, 1.0 };
const vec3 origin = { 0.0, 0.0, 0.0 };
const vec3 miss = { -1.0, -1.0, -1.0 };
const vec3 black = { 0.0, 0.0, 0.0 };
const vec3 white = { 1.0, 1.0, 1.0 };
const vec3 red = { 1.0, 0.0, 0.0 };
const vec3 green = { 0.0, 1.0, 0.0 };
const vec3 blue = { 0.0, 0.0, 1.0 };
const float _ignored = -1.0;

/****************************************************************************************************
 *
 * Material Definitions
 *
 ***************************************************************************************************/
const int material_type_invalid = -1;
const int material_type_diffuse = 0;
const int material_type_metallic = 1;
const int material_type_dielectric = 2;

struct material 
{
	// The primary color of a Lambertian material: must be divided by π so that it 
	// integrates to 1, as explained here: https://seblagarde.wordpress.com/tag/lambertian-surface/
	vec3 reflectance;

	// An integer denoting the type of the material (diffuse, metallic, etc.).
	int type;

	// `true` if this object is a light source and `false` otherwise.
	bool is_light;
};

const int light_index = 1;

material materials[] = 
{
	{ { 0.90, 0.80, 0.10 }, material_type_metallic, false }, 
	{ { 0.90, 0.10, 0.20 }, material_type_diffuse, false },
	{ { 0.968, 1.000, 0.968 }, material_type_diffuse, false }, 	// Off-white

	{ vec3(1.0, 0.35, 0.37), material_type_diffuse, false }, 	// Pink
    { vec3(0.54, 0.79, 0.15), material_type_diffuse, false }, 	// Mint
    { vec3(0.1, 0.51, 0.77), material_type_diffuse, false }, 	// Dark mint
	{ vec3(1.0, 0.79, 0.23), material_type_diffuse, false },	// Yellow
	{ vec3(0.42, 0.3, 0.58), material_type_diffuse, false }, 	// Purple

	{ vec3(8.0f, 8.0f, 8.0f), material_type_diffuse, true },  	// Light
};

/****************************************************************************************************
 *
 * Primitive Definitions
 *
 ***************************************************************************************************/
struct sphere
{
	float radius;
	vec3 center;
	int material_index;
};

struct plane
{
	vec3 normal;
	vec3 center;
	int material_index;
};

struct box
{
	vec3 min_pt;
	vec3 max_pt;
	int material_index;
};

struct quad 
{
	vec3 ul;
	vec3 ur;
	vec3 lr;
	vec3 ll;
	int material_index;
};

/****************************************************************************************************
 *
 * Scene Definition
 *
 ***************************************************************************************************/
sphere spheres[] = 
{
	{ 0.55, vec3( -1.4,  0.4, -1.3), 0 },
	{ 0.20, vec3(  1.0,  0.7, -1.6), 1 },
	{ 1.50, vec3(  0.0, -0.6,  0.0), 2 },
	{ 0.30, vec3( -1.4, -1.6, -1.5), 8 }
};

plane planes[] = 
{
	{ -z_axis,  z_axis * 3.5, 6 }, // Back
	{  z_axis, -z_axis * 4.5, 2 }, // Front
	{ -x_axis,  x_axis * 4.5, 3 }, // Left
	{  x_axis, -x_axis * 4.5, 4 }, // Right
	{  y_axis, -y_axis * 4.5, 7 }, // Top
	{ -y_axis,  y_axis * 1.0, 2 }  // Bottom
};

const int object_type_miss = -1;
const int object_type_sphere = 0;
const int object_type_plane = 1;
const int object_type_box = 2;
const int object_type_quad = 3; 

/****************************************************************************************************
 *
 * Utilities
 *
 ***************************************************************************************************/
float gpu_rnd(inout vec4 state) 
{
	// PRNG based on a weighted sum of four instances of the multiplicative linear congruential 
	// generator from the following paper: https://arxiv.org/pdf/1505.06022.pdf
	const vec4 q = vec4(1225, 1585, 2457, 2098);
	const vec4 r = vec4(1112, 367, 92, 265);
	const vec4 a = vec4(3423, 2646, 1707, 1999);
	const vec4 m = vec4(4194287, 4194277, 4194191, 4194167);

	vec4 beta = floor(state / q);
	vec4 p = a * mod(state, q) - beta * r;
	beta = (sign(-p) + vec4(1.0)) * vec4(0.5) * m;
	state = p + beta;

	return fract(dot(state / m, vec4(1.0, -1.0, 1.0, -1.0)));
}

vec3 palette(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d)
{
	// A wonderful function from Inigo Quilez for generating color 
	// palettes: http://iquilezles.org/www/articles/palettes/palettes.htm
    return a + b * cos(2.0 * pi * (c * t + d));
}

mat3 lookat(in vec3 from, in vec3 to)
{
	// This function constructs a look-at matrix that will orient a camera
	// positioned at `from` so that it is looking at `to`. First, 
	// we calculate the vector pointing from `from` to `to`. This
	// serves as the new z-axis in our camera's frame. Next, we take the 
	// cross-product between this vector and the world's y-axis to create
	// the x-axis in our camera's frame. Finally, we take the cross-product
	// between the two vectors that we just calculated in order to form 
	// the new y-axis in our camera's frame. 
	//
	// Together, these 3 vectors form the columns of our camera's look-at
	// (or view) matrix.
	vec3 camera_z = normalize(from - to);
	vec3 camera_x = cross(camera_z, y_axis);
	vec3 camera_y = cross(camera_x, camera_z);

	return mat3(camera_x, camera_y, camera_z);
}

vec3 cos_weighted_hemisphere(in vec3 normal, float rand_phi, float rand_radius) 
{    
	// Reference: `https://www.scratchapixel.com/lessons/3d-basic-rendering/global-illumination-path-tracing/global-illumination-path-tracing-practical-implementation`
	//
	// We can approximate the amount of light arriving at some point on the 
	// surface of a diffuse object by using Monte Carlo integration. Essentially,
	// we take the average of some number of unique samples of the function we 
	// are trying to approximate. Each sample must be divided by the the
	// PDF (probability density function) evaluated at the sample location.
	// For now, we are going to generate samples from a uniform distribution, 
	// meaning that every point on the hemisphere is just as likely to be
	// sampled as any other. This, in turn, means that the PDF is really just
	// a constant. But what is the value of this constant? It turns out after
	// a little bit of basic calculus, the PDF of our function is 1/2π every-
	// where! However, this PDF is expressed in terms of solid angles (a way
	// of measuring a portion of a sphere's surface area). What we'd really 
	// like is the PDF expressed in terms of polar coordinates θ and ϕ. 
	//
	// Why do we want this?  


	// The first thing we need to do is pick a random point on the surface
	// of the unit sphere. To do this, we use spherical coordinates, which
	// requires us to specify two angles: theta and phi. We choose each of 
	// these randomly.
	//
	// Then, we convert our spherical coordinates to cartesian coordinates
	// via the following formulae:
	//
	// 		x = r * cos(phi) * sin(theta)
	//		y = r * sin(phi) * sin(theta)
	//		z = r * cos(theta)
	//

	// Pick a random point on the unit disk and projeect it upwards onto 
	// the hemisphere. This generates the cosine-weighted distribution
	// that we are after.
	float radius = sqrt(rand_radius);
	float sample_x = radius * cos(2.0 * pi * rand_phi); 
	float sample_y = radius * sin(2.0 * pi * rand_phi);
	float sample_z = sqrt(1.0 - rand_radius);

    // In order to transform the sample from the world coordinate system
 	// to the local coordinate system around the surface normal, we need 
 	// to construct a change-of-basis matrix. 
 	//
 	// In this local coordinate system, `normal` is the y-axis. Next, we 
 	// need to find a vector that lies on the plane that is perpendicular
 	// to the normal. From the equation of a plane, we know that any vector
 	// `v` that lies on this tangent plane will satisfy the following 
 	// equation:
 	//
 	// 				dot(normal, v) = 0;
 	//
 	// We can assume that the y-coordinate is 0, leaving us with two 
 	// options:
 	//
 	// 1) x = N_z and z = -N_x
 	// 2) x = -N_z and z = N_x
    vec3 tangent = origin;
    if (abs(normal.x) > abs(normal.y))
    {
    	tangent = normalize(vec3(normal.z, 0.0, -normal.x));
    }
    else
    {
    	tangent = normalize(vec3(0.0, -normal.z, normal.y));
    }

    // Together, these three vectors form a basis for the local coordinate
 	// system. Note that there are more robust ways of constructing these
 	// basis vectors, but this is the simplest.
    vec3 local_x = normalize(cross(normal, tangent));
    vec3 local_y = cross(normal, local_x);
    vec3 local_z = normal;

	vec3 local_sample = vec3(sample_x * local_x + sample_y * local_y + sample_z * local_z);
    
    return normalize(local_sample);
}

float schlick(float cosine, float ior)
{
	float r0 = (1.0f - ior) / (1.0f + ior);
	r0 = r0 * r0;

	return r0 + (1.0f - r0) * pow(1.0f - cosine, 5.0f);
}

float max3(in vec3 e) 
{
  return max(max(e.x, e.y), e.z);
}

float min3(in vec3 e) 
{
  return min(min(e.x, e.y), e.z);
}

/****************************************************************************************************
 *
 * Intersection Routines
 *
 ***************************************************************************************************/
struct ray
{
	vec3 origin;
	vec3 direction;
};

vec3 get_point_at(in ray r, float t)
{
	// This is the parametric equation for a ray. Given a scalar value
	// `t`, it returns a point along the ray. We will use this throughout
	// the path tracing algorithm.
	return r.origin + r.direction * t;
}

bool intersect_sphere(in sphere sph, in ray r, out float t)
{
	vec3 oc = r.origin - sph.center;
	float a = dot(r.direction, r.direction);
	float b = dot(oc, r.direction);
	float c = dot(oc, oc) - sph.radius * sph.radius;

	float discriminant = b * b - a * c;

	// Avoid taking the square root of a negative number.
	if (discriminant < 0.0)
	{
		return false;
	}

	discriminant = sqrt(discriminant);
	float t0 = (-b + discriminant) / a;
	float t1 = (-b - discriminant) / a;

	// We want to take the smallest positive root. 
	if (t1 > epsilon)
	{
		t = t1;
	} 
	else if (t0 > epsilon)
	{
		t = t0;
	} 
	else 
	{	
		return false;
	}

	return true;
}

bool intersect_plane(in plane pln, in ray r, out float t)
{
	// We can write the equation for a plane, given its normal vector and
	// a single point `p` that lies on the plane. We can test if a different
	// point `v` is on the plane via the following formula:
	//
	//		dot(v - p, n) = 0
	// 
	// Now, we can substitute in the equation for our ray:
	//
	//		dot((o - t * d) - p, n) = 0
	//
	// Solving for `t` in the equation above, we obtain:
	//
	// 		numerator = dot(p - o, n)
	//		denominator = dot(d, n)
	//		t = numerator / denominator
	//
	// If the ray is perfectly parallel to the plane, then the denominator
	// will be zero. If the ray intersects the plane from behind, then `t`
	// will be negative. We want to avoid both of these cases.
	float denominator = dot(r.direction, pln.normal);
	float numerator = dot(pln.center - r.origin, pln.normal);

	// The ray is (basically) parallel to the plane.
	if (denominator > epsilon)
	{
		return false;
	}

	t = numerator / denominator;

	if (t > 0.0)
	{
		return true;
	}

	// The ray intersected the plane from behind.
	return false;
}

bool intersect_box(in box bx, in ray r, out float t)
{
	// TODO
	return false;
}

bool intersect_quad(in quad q, in ray r, out float t)
{	
	// See: `https://stackoverflow.com/questions/21114796/3d-ray-quad-intersection-test-in-java`
	const vec3 edge0 = q.ur - q.ul;
	const vec3 edge1 = q.ll - q.ul;
	const vec3 normal = normalize(cross(edge0, edge1));

	plane pln = plane(normal, vec3(q.ur), 0);

	float temp_t;
	if (intersect_plane(pln, r, temp_t))
	{
		vec3 m = r.origin + r.direction * temp_t;

		const float u = dot(m - q.ul, edge0);
		const float v = dot(m - q.ul, edge1);

		if (u >= epsilon && 
			v >= epsilon &&
			u <= dot(edge0, edge0) && 
			v <= dot(edge1, edge1))
		{
			t = temp_t;
			return true;	
		}
	}

	return false;
}

/****************************************************************************************************
 *
 * Tracing Routine
 *
 ***************************************************************************************************/

// An intersection holds several values:
// - The material index of the object that was hit (`material_index`)
// - The type of object that was hit (or a miss)
// - The direction vector of the `incident` ray
// - The location of intersection in world space (`position`)
// - The `normal` vector of the object that was hit, calculated at `position`
// - A scalar value `t`, which denotes a point along the incident ray
struct intersection 
{
	int material_index;
	int object_type;
	int object_index;
	vec3 incident;
	vec3 position;
	vec3 normal;
	float t;
};

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
	vec3 rand_hemi = normalize(cos_weighted_hemisphere(inter.normal, rand_a, rand_b));;
	vec3 reflected = reflect(inter.incident, inter.normal);

	// Diffuse is 0, metallic is 1.
	return mix(rand_hemi, reflected, mtl.type);
}

vec3 sample_sphere_light(inout vec4 seed, inout float cos_a_max, in vec3 light_position, in float radius, in vec3 ray_origin)
{
	 // Create a coordinate system for sampling
    const vec3 sw = normalize(light_position - ray_origin);
    const vec3 su = normalize(cross(abs(sw.x) > 0.01f ? y_axis : x_axis, sw));
    const vec3 sv = cross(sw, su);

    // Determine the max angle
    cos_a_max = sqrt(1.0f - (radius * radius) / dot(ray_origin - light_position, ray_origin - light_position));
    
    // Sample a sphere by solid angle
    const float eps1 = gpu_rnd(seed);
    const float eps2 = gpu_rnd(seed);
    const float cos_a = 1.0f - eps1 + eps1 * cos_a_max;
    const float sin_a = sqrt(1.0f - cos_a * cos_a);
    const float phi = two_pi * eps2;

    vec3 to_light = su * cos(phi) * sin_a + sv * sin(phi) * sin_a + sw * cos_a;

    return normalize(to_light);
}

vec2 random_on_disk(inout vec4 seed)
{
	// http://mathworld.wolfram.com/DiskPointPicking.html
	float radius = sqrt(gpu_rnd(seed));
	float theta = two_pi * gpu_rnd(seed);
	
	return vec2(radius * cos(theta), radius * sin(theta));
}

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
	
	vec3 offset = vec3(push_constants.cursor_position * 2.0f - 1.0f, 0.0f) * 8.0f;
	vec3 camera_position = vec3(0.0, -3.0, -8.5) + offset;

	const float vertical_fov = 45.0f;
	const float aperture = 0.5f;
	const float lens_radius = aperture / 2.0f;
	const float theta = vertical_fov * pi / 180.0f;
	const float half_height = tan(theta * 0.5f);
	const float half_width = aspect_ratio * half_height;

	mat3 look_at = lookat(camera_position, origin);

	float dist_to_focus = push_constants.cursor_position.x * 5.0 + 5.0;
	dist_to_focus = length(camera_position);

	vec3 lower_left_corner = camera_position - look_at * vec3(half_width, half_height, 1.0) * dist_to_focus;
	vec3 horizontal = 2.0 * half_width * dist_to_focus * look_at[0];
	vec3 vertical = 2.0 * half_height * dist_to_focus * look_at[1];

    const vec3 light_position = { -1.0f, -5.0f, 1.0f };
    const float light_radius = 0.75f;
	
	for (uint j = 0; j < number_of_iterations; ++j)
	{
		// By jittering the uv-coordinates a tiny bit here, we get 
		// "free" anti-aliasing.
		vec2 jitter = { gpu_rnd(seed), gpu_rnd(seed) };
		jitter = jitter * 2.0 - 1.0;
		uv += (jitter / push_constants.resolution) * anti_aliasing;
		
		// Depth-of-field calculation.
		vec3 rd = lens_radius * vec3(random_on_disk(seed), 0.0);
		vec3 lens_offset = look_at * vec3(rd.xy, 0.0);
		vec3 ro = camera_position + lens_offset;
		rd = lower_left_corner + uv.x * horizontal + uv.y * vertical - camera_position - lens_offset;

		// Calculate the ray direction based on the current fragment's
		// uv-coordinates. All rays will originate from the camera's
		// location. 		
		ray r = { ro, rd };

		// Define some colors.
		const vec3 sky = black;
		vec3 radiance = black;
		vec3 throughput = white;

		int prev_material_type = 0;

		// This is the main path tracing loop.
		for (uint i = 0; i < number_of_bounces; ++i)
		{	
			intersection inter = intersect_scene(r);

            // There were no intersections: simply accumulate the background color and break
            if (inter.object_type == object_type_miss) 
            {
                radiance += throughput * sky;   
                break;
            }
            // There was an intersection: accumulate color and bounce
            else 
            {
            	material mtl = materials[inter.material_index];

                const vec3 hit_location = r.origin + r.direction * inter.t;

                // When using explicit light sampling, we have to account for a number of edge cases:
                //
                // 1. If this is the first bounce, and the object we hit is a light, we need to add its
                //    color (otherwise, lights would appear black in the final render)
                // 2. If the object we hit is a light, and the PREVIOUS object we hit was specular (a
                //    metal or dielectric), we need to add its color (otherwise, lights would appear
                //    black in mirror-like objects)
                if ((j == 0 || 
                    prev_material_type == material_type_dielectric || 
                    prev_material_type == material_type_metallic) && 
                    mtl.is_light) 
                {
                    radiance += throughput * mtl.reflectance;   
                }
            
                // Set the new ray origin
                r.origin = hit_location + inter.normal * epsilon;

                // Choose a new ray direction based on the material type
                if (mtl.type == material_type_diffuse)
                {

                    // Sample all of the light sources (right now we only have one)
                    // ...

                    float cos_a_max = 0.0f;
                    const vec3 light_position = spheres[2].center;
                    const float light_radius = spheres[2].radius;
                    vec3 to_light = sample_sphere_light(seed, cos_a_max, light_position, light_radius, r.origin);

                    ray secondary_ray = { r.origin, to_light };

                    intersection secondary_inter = intersect_scene(secondary_ray);
                    material secondary_mtl = materials[secondary_inter.material_index];

                    const float light_id = 3.0f;

                    if (secondary_inter.t > 0.0f &&      						// We hit an object
                        secondary_inter.object_type == object_type_sphere &&    
                        secondary_mtl.is_light && 								// ...and that object was the light source 
                        !mtl.is_light)      									// ...and the original object wasn't the light source (avoid self-intersection)  
                    {
                        const float omega = (1.0f - cos_a_max) * two_pi;
                        const vec3 normal_towards_light = dot(inter.normal, r.direction) < 0.0f ? inter.normal : -inter.normal;

                        const float pdf = max(0.0f, dot(to_light, normal_towards_light)) * omega * (1.0f / pi);

                        radiance += throughput * (vec3(1.0, 0.98, 0.98) * 6.0f) * pdf; 
                    }

                    r.direction = normalize(cos_weighted_hemisphere(inter.normal, gpu_rnd(seed), gpu_rnd(seed)));

                    // Accumulate material color
                    const float cos_theta = max(0.0f, dot(inter.normal, r.direction));
                    const float pdf = 1.0f / two_pi;
                    const vec3 brdf = mtl.reflectance * (1.0f / pi);

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

	if (push_constants.mouse_down == 1.0)
	{
		o_color = vec4(curr_frame, 1.0);
	}
	else
	{
		o_color = vec4(prev_frame + curr_frame, 1.0);
	}
}
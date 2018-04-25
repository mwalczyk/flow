#version 450
#extension GL_ARB_separate_shader_objects : enable

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
const float pi = 3.1415926535897932384626433832795;
const float gamma = 1.0 / 2.2;
const float anti_aliasing = 0.75;
const uint number_of_iterations = 10;
const uint number_of_bounces = 5;
const float epsilon = 0.001;
const float max_distance = 10000.0;

/****************************************************************************************************
 *
 * Vector Constants
 *
 ***************************************************************************************************/
const vec3 x_axis = vec3(1.0, 0.0, 0.0);
const vec3 y_axis = vec3(0.0, 1.0, 0.0);
const vec3 z_axis = vec3(0.0, 0.0, 1.0);
const vec3 origin = vec3(0.0);
const vec3 miss = vec3(-1.0);
const vec3 black = vec3(0.0);
const vec3 white = vec3(1.0);
const vec3 red = vec3(1.0, 0.0, 0.0);
const vec3 green = vec3(0.0, 1.0, 0.0);
const vec3 blue = vec3(0.0, 0.0, 1.0);

/****************************************************************************************************
 *
 * Material Definitions
 *
 ***************************************************************************************************/
const float _ignored = -1.0;

const int material_type_diffuse = 0;
const int material_type_metallic = 1;
const int material_type_dielectric = 2;

struct material 
{
	// The primary color of the material
	vec3 alebdo;

	// The "glossiness" of the material, where a value of 0.0 is full roughness. Note
	// that this parameter only affects materials of type `material_type_metallic`.
	float roughness;

	// The index of refraction of the material. Note that this parameter only affects
	// materials of type `material_type_dielectric`.
	float ior;

	// An integer denoting the type of the material (diffuse, metallic, etc.).
	int type;
};

material materials[] = {
	material(white, _ignored, _ignored, material_type_diffuse)
};

/****************************************************************************************************
 *
 * Primitive Definitions
 *
 ***************************************************************************************************/

// NOTE: There are definitely more elegant ways to handle materials here -
// I just wanted to keep this part simple for now.
//
// A sphere has a radius, center position, and color (albedo).
struct sphere
{
	float radius;
	vec3 center;
	vec3 albedo;
};

// A plane has a normal, center position, and color (albedo).
struct plane
{
	vec3 normal;
	vec3 center;
	vec3 albedo;
};

/****************************************************************************************************
 *
 * Scene Definition
 *
 ***************************************************************************************************/
const uint number_of_spheres = 4;
const uint number_of_planes = 4;

// Some spheres on the ground.
sphere spheres[] = 
{
	sphere(100.0, vec3(0.0, 100.9, 0.0), vec3(1.00, 0.80, 0.80)), // Ground
	sphere(0.55, vec3(-1.4,  0.4, -1.3), vec3(0.90, 0.80, 0.10)),
	sphere(0.20, vec3( 1.0,  0.7, -1.6), vec3(0.90, 0.10, 0.20)),
	sphere(1.50, vec3( 0.0, -0.6,  0.0), vec3(0.75, 0.75, 0.60))
};

// Create the walls of our scene.
plane planes[] = 
{
	plane(-z_axis,  z_axis * 3.5, white * 0.4), // Back
	plane( z_axis, -z_axis * 4.5, white * 0.4), // Front
	plane(-x_axis,  x_axis * 4.5, white * 0.4), // Left
	plane( x_axis, -x_axis * 4.5, white * 0.4), // Right
};

/****************************************************************************************************
 *
 * Utilities
 *
 ***************************************************************************************************/
float rand_stable(in vec2 seed)
{
	// There are a lot of ways to generate random numbers in GLSL, but this
	// one-liner is the one that I am most familiar with. See the S.O. post
	// here: https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
	return fract(sin(dot(seed, vec2(12.9898, 78.233))) * 43758.5453);
}

float rand_dynamic(in vec2 seed) 
{	
	// This is the same as the function above, except we change the random 
	// values over time by adding the uniform time.
    return fract(sin(dot(seed, vec2(12.9898, 78.233) + push_constants.time)) * 43758.5453);
}

vec3 palette(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d)
{
	// A wonderful function from Inigo Quilez for generating color 
	// palettes: http://iquilezles.org/www/articles/palettes/palettes.htm
    return a + b * cos(2.0 * pi * (c * t + d));
}

mat3 lookat(in vec3 view_point, in vec3 target)
{
	// This function constructs a look-at matrix that will orient a camera
	// positioned at `view_point` so that it is looking at `target`. First, 
	// we calculate the vector pointing from `view_point` to `target`. This
	// serves as the new z-axis in our camera's frame. Next, we take the 
	// cross-product between this vector and the world's y-axis to create
	// the x-axis in our camera's frame. Finally, we take the cross-product
	// between the two vectors that we just calculated in order to form 
	// the new y-axis in our camera's frame. 
	//
	// Together, these 3 vectors form the columns of our camera's look-at
	// (or view) matrix.
	vec3 camera_z = normalize(view_point - target);
	vec3 camera_x = cross(camera_z, y_axis);
	vec3 camera_y = cross(camera_x, camera_z);

	return mat3(camera_x, camera_y, camera_z);
}

vec3 hemisphere(in vec3 normal, in vec2 seed) 
{
	// Explained here: https://www.scratchapixel.com/lessons/3d-basic-rendering/global-illumination-path-tracing/global-illumination-path-tracing-practical-implementation
	const float offset = 100.0;

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
    float z = rand_dynamic(seed);
    float sin_theta = sqrt(1.0 - z * z);
    float phi = 2.0 * pi * rand_dynamic(seed + offset);
    float x = cos(phi) * sin_theta;
    float y = sin(phi) * sin_theta;

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
 	//
 	// Here, we choose option (1). So, a vector in the tangent plane is:
 	// <N_z, 0, -N_x>. By taking the cross product between `normal` and
 	// this new vector, we can obtain a third vector that is orthogonal
 	// to both.
 	//
 	// Together, these three vectors form a basis for the local coordinate
 	// system. Note that there are more robust ways of constructing these
 	// basis vectors, but this is the simplest.
    vec3 tangent = normalize(vec3(normal.z, 0.0, -normal.x));

    // Transform the sample from world to local coordinates.
    vec3 local_x = normalize(cross(normal, tangent));
    vec3 local_y = cross(normal, local_x);
    vec3 local_z = normal;

    return normalize(local_x * x + local_y * y + local_z * z);
}

/****************************************************************************************************
 *
 * Intersection Routines
 *
 ***************************************************************************************************/
 
 // A ray has an origin and direction, which should be a unit vector.
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
	vec3 temp = r.origin - sph.center;
	float b = 2.0 * dot(temp, r.direction);
	float c = dot(temp, temp) - sph.radius * sph.radius;

	float discriminant = b * b - 4.0 * c;

	// Avoid taking the square root of a negative number.
	if (discriminant < 0.0)
	{
		return false;
	}

	discriminant = sqrt(discriminant);
	float t0 = -b + discriminant;
	float t1 = -b - discriminant;

	// We want to take the smallest positive root. 
	if (t1 > epsilon)
	{
		t = t1 * 0.5;
	} 
	else if (t0 > epsilon)
	{
		t = t0 * 0.5;
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

/****************************************************************************************************
 *
 * Materials
 *
 ***************************************************************************************************/
vec3 scatter_diffuse(in vec3 point, in vec3 incident, in vec3 normal, in vec2 seed)
{
	vec3 rand_hemi = normalize(hemisphere(normal, seed));

	return rand_hemi;
}

vec3 scatter_metallic(in vec3 point, in vec3 incident, in vec3 normal, in vec2 seed, in float roughness)
{
	vec3 rand_hemi = normalize(hemisphere(normal, seed));
	vec3 reflected = reflect(incident, normal);

	return mix(rand_hemi, reflected, roughness);
}

vec3 scatter_dielectric(in vec3 point, in vec3 incident, in vec3 normal, in vec2 seed, in float ior)
{
	// TODO
	return miss;
}

vec3 scatter(in ray r, in material mat)
{
	// TODO
	return miss;
}


/****************************************************************************************************
 *
 * Tracing Routine
 *
 ***************************************************************************************************/

// In the `intersection` struct below, the `object_type` field will be one of the following: 
const int object_type_miss = -1;
const int object_type_sphere = 0;
const int object_type_plane = 1; 

// An intersection holds several values:
// - A numeric identifier describing the type of object that was hit (`object_type`)
// - A numeric identifier describing the index of the object that was hit (`object_index`)
// - The location of intersection in world space (`position`)
// - The `normal` vector of the object that was hit, calculated at `position`
// - A scalar value `t`, which denotes a point along the incident ray
struct intersection 
{
	int object_type;
	int object_index;
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
	intersection inter;
	inter.object_type = object_type_miss;
	inter.object_index = -1;
	inter.t = max_distance;

	// Now, let's iterate through the objects in our scene, starting with 
	// the spheres:
	for(int i = 0; i < number_of_spheres; ++i)
	{	
		// Did the ray intersect this object?
		float temp_t;
		if (intersect_sphere(spheres[i], r, temp_t))
		{
			// Was the intersection closer than any previous one?
			if (temp_t < inter.t)
			{
				inter.object_type = object_type_sphere;
				inter.object_index = i;
				inter.position = r.origin + r.direction * temp_t;
				inter.normal = normalize(inter.position - spheres[i].center);
				inter.t = temp_t;
			}
		} 
	}

	// then planes:
	for(int i = 0; i < number_of_planes; ++i)
	{	
		// Did the ray intersect this object?
		float temp_t;
		if (intersect_plane(planes[i], r, temp_t))
		{
			// Was the intersection closer than any previous one?
			if (temp_t < inter.t)
			{
				inter.object_type = object_type_plane;
				inter.object_index = i;
				inter.position = r.origin + r.direction * temp_t;
				inter.normal = normalize(planes[i].normal);
				inter.t = temp_t;
			}
		} 
	}

	return inter;
}

vec3 trace()
{
	// We want to make sure that we correct for the window's aspect ratio
	// so that our final render doesn't look skewed or stretched when the
	// resolution changes.
	float aspect_ratio = push_constants.resolution.x / push_constants.resolution.y;
	vec2 uv = (gl_FragCoord.xy / push_constants.resolution) * 2.0 - 1.0;
	uv.x *= aspect_ratio;

	vec3 final = black;

	for (uint j = 0; j < number_of_iterations; ++j)
	{
		// By jittering the uv-coordinates a tiny bit here, we get 
		// "free" anti-aliasing.
		vec2 jitter = vec2(rand_dynamic(uv + j), rand_dynamic(uv + j + 100.0)) * 2.0 - 1.0;
		uv += (jitter / push_constants.resolution) * anti_aliasing;

		// Calculate the ray direction based on the current fragment's
		// uv-coordinates. All rays will originate from the camera's
		// location. 
		vec3 offset = vec3(push_constants.cursor_position * 2.0 - 1.0, 0.0) * 4.0;
		vec3 camera_position = vec3(0.0, -2.0, -4.0) + offset;
		vec3 ro = camera_position;
		vec3 rd = normalize(lookat(origin, ro) * vec3(uv, 1.0));
		ray r = ray(ro, rd);

		// Define some colors.
		vec3 sky = black;
		vec3 color = black;
		vec3 accumulated = white;

		// This is the main path tracing loop.
		for (uint i = 0; i < number_of_bounces; ++i)
		{	
			intersection itr = intersect_scene(r);
			vec3 incident = r.direction;

			// Per-bounce random seed
			const vec2 seed = gl_FragCoord.xy + i + j * 100.0;

			vec3 bounce = scatter_diffuse(itr.position, r.direction, itr.normal, seed);
			r.origin = itr.position + itr.normal * epsilon;
			r.direction = bounce;

			// Based on the type of object that was hit, we can choose how to
			// react.
			switch(itr.object_type)
			{
			case object_type_sphere:

				if (itr.object_index == 1)
				{
					r.direction = scatter_metallic(itr.position, incident, itr.normal, seed, 1.0);
				}

				accumulated *= 2.0 * spheres[itr.object_index].albedo;
				break;

			case object_type_plane:
				
				// The back plane is emissive
				if (itr.object_index == 0) 
				{	
					float pct = step(1.0, mod(itr.position.x * 2.0 + 0.5, 2.0)); 
					vec3 emissive = palette(itr.position.x * 0.1, 
											vec3(0.5, 0.5, 0.5), 
											vec3(0.5, 0.5, 0.5), 
											vec3(1.0, 1.0, 1.0), 
											vec3(0.00, 0.33, 0.67));
					
					color += white * accumulated * pct;
				}

				accumulated *= 2.0 * planes[itr.object_index].albedo;
				break;

			case object_type_miss:

				color += accumulated * sky;
				break;
			}
		}

		final += color;
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
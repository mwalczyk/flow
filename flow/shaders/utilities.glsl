const float pi = 3.1415926535897932384626433832795f;
const float two_pi = pi * 2.0f;
const float one_over_pi = 1.0f / pi;
const float one_over_two_pi = 1.0f / two_pi;

const float gamma = 1.0f / 2.2f;
const float anti_aliasing = 0.5f;
const uint number_of_iterations = 1;
const uint number_of_bounces = 10;
const float epsilon = 0.001f;
const float min_distance = 0.0f;
const float max_distance = 100.0f;

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

float map(float v, float in_min, float in_max, float out_min, float out_max) 
{
  return out_min + (out_max - out_min) * (v - in_min) / (in_max - in_min);
}

float to_radians(float degrees)
{
	return degrees * (pi / 180.0f);
}

mat4 rotation(in vec3 axis, float angle)
{
	// From Neil Mendoza
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0f - c;
    
    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0f,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0f,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0f,
                0.0f,                               0.0f,                               0.0f,                               1.0f);
}

vec3 palette(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d)
{
	// From IQ
    return a + b * cos(two_pi * (c * t + d));
}

vec3 rainbow(in float t)
{
	return palette(t, vec3(0.5f), vec3(0.5f), white, vec3(0.0f, 0.33f, 0.67f));
}


float mod289(float x) 
{ 
	return x - floor(x * (1.0 / 289.0)) * 289.0; 
}

vec4 mod289(in vec4 x) 
{ 
	return x - floor(x * (1.0 / 289.0)) * 289.0; 
} 

vec4 perm(in vec4 x) 
{ 
	return mod289(((x * 34.0) + 1.0) * x); 
}

float noise(in vec3 p)
{
	// From Morgan McGuire ("Graphics Codex")
    vec3 a = floor(p);
    vec3 d = p - a;
    d = d * d * (3.0 - 2.0 * d);

    vec4 b = a.xxyy + vec4(0.0, 1.0, 0.0, 1.0);
    vec4 k1 = perm(b.xyxy);
    vec4 k2 = perm(k1.xyxy + b.zzww);

    vec4 c = k2 + a.zzzz;
    vec4 k3 = perm(c);
    vec4 k4 = perm(c + 1.0);

    vec4 o1 = fract(k3 * (1.0 / 41.0));
    vec4 o2 = fract(k4 * (1.0 / 41.0));
    vec4 o3 = o2 * d.z + o1 * (1.0 - d.z);
    vec2 o4 = o3.yw * d.x + o3.xz * (1.0 - d.x);

    return o4.y * d.y + o4.x * (1.0 - d.y);
}

float rand(in vec2 seed)
{
    return fract(sin(dot(seed.xy, vec2(12.9898f, 78.233f))) * 43758.5453f);
}

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

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

#define pi 3.1415926535897932384626433832795
#define PI2 6.283185307179586476925286766559
#define PI1 0.31830988618379067153776752674503
#define PI180 0.01745329251994329576923690768489
#define gamma 0.45454545454545454545454545454545

/*
const float pi = 3.1415926535897932384626433832795;
const float PI2 = pi + pi;
const float PI1 = 1.0 / pi;
const float PI180 = pi / 180.0;
const float gamma = 1.0 / 2.2;
*/

const float anti_aliasing = 0.55;
const uint number_of_iterations = 1;
const uint number_of_bounces = 4;
const float epsilon = 0.001;
const float max_distance = 10000.0;



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
const int material_type_metallic = 1;	// <= mirror
const int material_type_dielectric = 2;	// <= glass
const int material_type_emissive = 3;

struct material 
{
	// The primary color of a Lambertian material: must be divided by π so that it 
	// integrates to 1, as explained here: https://seblagarde.wordpress.com/tag/lambertian-surface/
	vec3 reflectance;

	// An integer denoting the type of the material (diffuse, metallic, etc.).
	int type;
};

material materials[] = 
{
	{ { 0.90, 0.80, 0.10 }, material_type_metallic }, 
	{ { 0.90, 0.10, 0.20 }, material_type_diffuse },
	{ { 0.968, 1.000, 0.968 }, material_type_diffuse }, // Off-white

	{ vec3(1.0, 0.35, 0.37), material_type_diffuse }, // Pink
    { vec3(0.54, 0.79, 0.15), material_type_diffuse }, // Mint
    { vec3(0.1, 0.51, 0.77), material_type_diffuse }, // Dark mint
	{ vec3(1.0, 0.79, 0.23), material_type_diffuse },	// Yellow
	{ vec3(0.42, 0.3, 0.58), material_type_diffuse }, // Purple
};

/****************************************************************************************************
 *
 * Primitive Definitions
 *
 ***************************************************************************************************/
struct sphere
{
	float radius, r2;
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
};

struct area_light 
{
	vec3 ul;
	vec3 ur;
	vec3 lr;
	vec3 ll;
	vec3 intensity;
	vec3 normal;
	float area;
	float dot_edge0;
	float dot_edge1;
};

/****************************************************************************************************
 *
 * Scene Definition
 *
 ***************************************************************************************************/
sphere spheres[] = 
{
	{ 0.55, 0.55*0.55, vec3(-1.4,  0.4, -1.3), 0 },
	{ 0.20, 0.20*0.20, vec3( 1.0,  0.7, -1.6), 1 },
	{ 1.50, 1.50*1.50, vec3( 0.0, -0.6,  0.0), 2 }
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

area_light scene_light = 
{
	{ -3.0, -3.5,  0.0,  }, // ul
	{  3.0, -3.5,  0.0,  }, // ur
	{  3.0, -3.5, -3.0,  }, // lr
	{ -3.0, -3.5, -3.0,  }, // ll
	white,
	white, //normal
	0.0,
	0.0,
	0.0
};

/****************************************************************************************************
 *
 * Utilities
 *
 ***************************************************************************************************/
// PRNG based on a weighted sum of four instances of the multiplicative linear congruential 
// generator from the following paper: https://arxiv.org/pdf/1505.06022.pdf


	const vec4 q = vec4(1225, 1585, 2457, 2098);
	const vec4 r = vec4(1112, 367, 92, 265);
	const vec4 a = vec4(3423, 2646, 1707, 1999);
	const vec4 m = vec4(4194287, 4194277, 4194191, 4194167);
	
	const vec4 m1 = vec4(1.0) / m;
	const vec4 q1 = vec4(1.0) / q;


float gpu_rnd(inout vec4 state) 
{

	vec4 beta = floor(state * q1);
	vec4 p = a * mod(state, q) - beta * r;
	beta = (sign(-p) + vec4(1.0)) * vec4(0.5) * m;
	state = p + beta;

	return fract(dot(state * m1, vec4(1.0, -1.0, 1.0, -1.0)));
}

vec3 palette(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d)
{
	// A wonderful function from Inigo Quilez for generating color 
	// palettes: http://iquilezles.org/www/articles/palettes/palettes.htm
    return a + b * cos(PI2 * (c * t + d));
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
	// Reference: `// Explained here: https://www.scratchapixel.com/lessons/3d-basic-rendering/global-illumination-path-tracing/global-illumination-path-tracing-practical-implementation`
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
	float sample_x = radius * cos(PI2 * rand_phi); 
	float sample_y = radius * sin(PI2 * rand_phi);
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
	
	/*
    vec3 tangent = origin;
    if (abs(normal.x) > abs(normal.y))
    {
    	tangent = 
		//normalize
		(vec3(normal.z, 0.0, -normal.x));
    }
    else
    {
    	tangent = 
		//normalize
		(vec3(0.0, -normal.z, normal.y));
    }

    // Together, these three vectors form a basis for the local coordinate
 	// system. Note that there are more robust ways of constructing these
 	// basis vectors, but this is the simplest.
    vec3 local_x = 
	//normalize
	(cross(normal, tangent));
    vec3 local_y = cross(normal, local_x);
    vec3 local_z = normal;
	*/
	
	vec3 u, w;
	 
	if (normal.z < 0.0)
	{	
		float a = 1.0 / (1.0 - normal.z); 
		float b = normal.x * normal.y * a;
		u = vec3(1.0 - normal.x * normal.x * a, -b, normal.x); 
		w = vec3(b, normal.y * normal.y * a - 1.0, -normal.y);
	}
	else
	{
		float a = 1.0 / (1.0 + normal.z); 
		float b = -normal.x * normal.y * a;
		u = vec3(1.0 - normal.x * normal.x * a, b, -normal.x); 
		w = vec3(b, 1.0 - normal.y * normal.y * a, -normal.y);
	}
	
	vec3 local_sample = vec3(sample_x * u + sample_y * w + sample_z * normal);
	
    return 
	//normalize
	(local_sample);
}

float total_area(in area_light light)
{
	vec3 edge0 = light.ur - light.ul;
	vec3 edge1 = light.ll - light.ul;

	return length(edge0) * length(edge1);
}

vec3 generate_sample_on_light(in area_light light, float rand_a, float rand_b)
{
	vec3 edge0 = light.ur - light.ul;
	vec3 edge1 = light.ll - light.ul;

	vec3 pt = edge0 * rand_a + edge1 * rand_b;
	pt.y = -3.5;
	
	return pt;
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

/*
double intersect(const Ray &r) const { // returns distance, 0 if nohit
Vec op = p-r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
double t, eps=1e-4, b=op.dot(r.d), det=b*b-op.dot(op)+rad2;
if (det<0) return 0; else det=sqrt(det);
return (t=b-det)>eps ? t : ((t=b+det)>eps ? t : 0);
}
*/

bool intersect_sphere(in sphere sph, in ray r
//, in float a, in float aa
, out float t)
{
	vec3 op = sph.center - r.origin;
	float b = dot(op, r.direction);
	//float c = dot(temp, temp) - sph.r2;
	
	float discriminant = b * b - dot(op, op) + sph.r2;  //4*dot*dot - 4*c = 2*2*(dot*dot - c)
	//7* vs 9*, consts
	
	if (discriminant < 0.0)
	{
		return false;
	}
	
	discriminant = sqrt(discriminant);
	
	float t0 = b + discriminant;
	float t1 = b - discriminant;
	
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
		
	
	/*
	vec3 oc = r.origin - sph.center;
	//float a = dot(r.direction, r.direction);
	float b = dot(oc, r.direction);
	float c = dot(oc, oc) - sph.r2;

	float discriminant = b * b - a * c;

	// Avoid taking the square root of a negative number.
	if (discriminant < 0.0)
	{
		return false;
	}

	discriminant = sqrt(discriminant);
	//float t1 = (-b - discriminant) / a;
	float t1 = (-b - discriminant) * aa;

	// We want to take the smallest positive root. 
	if (t1 > epsilon)
	{
		t = t1;
	} 
	else 
	{
		//t1 = (-b + discriminant) / a;
		t1 = (-b + discriminant) * aa;
		if (t1 > epsilon)
		{
			t = t1;
		}
		else 
		{	
			return false;
		}
	}
	*/

	return true;
}

bool intersect_plane(in plane pln, in ray r, out float t)
{
	//r.direction = normalize(r.direction);

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

	// vd > -EPS && vd   < EPS
	// The ray is (basically) parallel to the plane.
	if (
	denominator > epsilon
	)
	{
		return false;
	}
	
	float numerator = dot(pln.center - r.origin, pln.normal);

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
	return false;
}

bool intersect_area_light(in area_light light, in ray r, out float t)
{	
	//r.direction = normalize(r.direction);

	// See: https://stackoverflow.com/questions/21114796/3d-ray-quad-intersection-test-in-java
	/*
	vec3 edge0 = light.ur - light.ul;
	vec3 edge1 = light.ll - light.ul;
	vec3 normal = 
	//normalize
	(cross(edge0, edge1));

	plane pln = plane(normal, vec3(light.ur), 0);	
	*/
	
	plane pln = plane(light.normal, vec3(light.ur), 0);

	float temp_t;
	if (intersect_plane(pln, r, temp_t)) //2dot, 1/
	{
		//6* 1/
		vec3 m = r.origin + r.direction * temp_t; //3*

		float u = dot(m - light.ul, light.ur - light.ul);
		if (
			u < epsilon ||
			u > light.dot_edge0
			)
		{
			//12* 1/
			return false;
		}
		
		float v = dot(m - light.ul, light.ll - light.ul);
		//2dot
		//= 4dot 3* 1/ => 15* 1/
		//very good rect intersection code

		/*
		if (u >= epsilon && v >= epsilon &&
			u <= light.dot_edge0 && v <= light.dot_edge1)
			*/
		if (v >= epsilon &&
			v <= light.dot_edge1)
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

// In the `intersection` struct below, the `object_type` field will be one of the following: 
const int object_type_miss = -1;
const int object_type_sphere = 0;
const int object_type_plane = 1; 
const int object_type_area_light = 2;

// An intersection holds several values:
// - The material index of the object that was hit (`material_index`)
// - The type of object that was hit (or a miss)
// - The direction vector of the incident ray
// - The location of intersection in world space (`position`)
// - The `normal` vector of the object that was hit, calculated at `position`
// - A scalar value `t`, which denotes a point along the incident ray
struct intersection 
{
	int material_index;
	int object_type;
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
		r.direction,
		r.origin,
		{ -1.0, -1.0, -1.0 },
		max_distance
	};

	
	//float a = dot(r.direction, r.direction);
	//float aa = 1.0/a;
	
	// Now, let's iterate through the objects in our scene, starting with 
	// the spheres:
	for(int i = 0; i < spheres.length(); ++i)
	{	
		// Did the ray intersect this object?
		float temp_t;
		if (intersect_sphere(spheres[i], r, 
		// a, aa, 
		temp_t))
		{
			// Was the intersection closer than any previous one?
			if (temp_t < inter.t)
			{
				inter.material_index = spheres[i].material_index;
				inter.object_type = object_type_sphere;
				inter.position = r.origin + r.direction * temp_t;
				inter.normal = 
				normalize
				(inter.position - spheres[i].center);
				inter.t = temp_t;
			}
		} 
	}

	// then planes:
	for(int i = 0; i < planes.length(); ++i)
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
				inter.position = r.origin + r.direction * temp_t;
				inter.normal = 
				//normalize
				(planes[i].normal);
				inter.t = temp_t;
			}
		} 
	}

	// and our single light source:
	float temp_t;
	if (intersect_area_light(scene_light, r, temp_t))
	{
		if (temp_t < inter.t)
		{
			inter.material_index = -1;
			inter.object_type = object_type_area_light;
			inter.position = r.origin + r.direction * temp_t;
			inter.normal = -z_axis;
			inter.t = temp_t;
		}
	}

	return inter;
}

/*
vec3 scatter(in material mtl, in intersection inter, float rand_a, float rand_b)
{
	//vec3 r1 = normalize(cos_weighted_hemisphere(inter.normal, rand_a, rand_b));
	//vec3 r2 = normalize(hemisphere(inter.normal, rand_a, rand_b));
	//vec3 r = mix(r1, r2, sin(push_constants.time) * 0.5 + 0.5);
	
	if(mtl.type == material_type_metallic)
		return reflect(inter.incident, inter.normal);
	else return 
	//normalize
	(cos_weighted_hemisphere(inter.normal, rand_a, rand_b));;
	//vec3 rand_hemi = normalize(hemisphere(inter.normal, rand_a, rand_b));

	// Diffuse is 0, metallic is 1.
	//return mix(rand_hemi, reflected, mtl.type);
}
*/

/*
vec3 scatter(in material mtl, in intersection inter, float rand_a, float rand_b)
{
	//vec3 r1 = normalize(cos_weighted_hemisphere(inter.normal, rand_a, rand_b));
	//vec3 r2 = normalize(hemisphere(inter.normal, rand_a, rand_b));
	//vec3 r = mix(r1, r2, sin(push_constants.time) * 0.5 + 0.5);

	vec3 rand_hemi = 
	//normalize
	(cos_weighted_hemisphere(inter.normal, rand_a, rand_b));;
	//vec3 rand_hemi = normalize(hemisphere(inter.normal, rand_a, rand_b));
	vec3 reflected = reflect(inter.incident, inter.normal);

	// Diffuse is 0, metallic is 1.
	return mix(rand_hemi, reflected, mtl.type);
}
*/



vec3 sample_light_source(in vec3 position, in vec3 normal, in vec3 brdf, inout vec4 seed)
{
	// See: http://www.cs.uu.nl/docs/vakken/magr/2015-2016/slides/lecture%2008%20-%20variance%20reduction.pdf
	float rand_u = gpu_rnd(seed);
	float rand_v = gpu_rnd(seed);
	vec3 position_on_light_source = generate_sample_on_light(scene_light, rand_u, rand_v);
	vec3 normal_of_light_source = y_axis;

	vec3 to_light_source = 
	//normalize
	(position_on_light_source - position);

	float falloff_at_light = dot(normal_of_light_source, -to_light_source);
	float falloff_at_current_point = dot(normal, to_light_source);

	if (falloff_at_light > 0.0 && falloff_at_current_point > 0.0)
	{
		
		float dist = distance(position, position_on_light_source);
		
		ray r = { position, to_light_source/dist };
		intersection inter = intersect_scene(r);

		// Check for occlusions.
		if (inter.object_type == object_type_area_light)
		{
			//float area = total_area(scene_light);
			float area = scene_light.area;
			
			
			float solid_angle = (falloff_at_light * area) / (dist * dist * dist * dist);
			
			
			//float solid_angle = (falloff_at_light * area) / length(position_on_light_source - position);
			//float solid_angle = (falloff_at_light * area) / length( position - position_on_light_source);

			return scene_light.intensity * solid_angle * brdf * falloff_at_current_point;
		}

	}

	return black;
}



vec2 random_on_disk(inout vec4 seed)
{
	// http://mathworld.wolfram.com/DiskPointPicking.html
	float radius = sqrt(gpu_rnd(seed));
	float theta = PI2 * gpu_rnd(seed);
	
	return vec2(radius * cos(theta), radius * sin(theta));
}

vec3 trace()
{
	// We want to make sure that we correct for the window's aspect ratio
	// so that our final render doesn't look skewed or stretched when the
	// resolution changes.
	float aspect_ratio = push_constants.resolution.x / push_constants.resolution.y;
	vec2 uv = gl_FragCoord.xy / push_constants.resolution;
	
	/*
	vec2 uv = (gl_FragCoord.xy / push_constants.resolution) * 2.0 - 1.0;
	uv.x *= aspect_ratio;
	*/
	
	float t = push_constants.time;
	vec4 seed = { uv.x + t * 41.13, 
	              uv.y + t * 113.0, 
	              uv.x - t * 7.57, 
	              uv.y - t * 67.0 };
	
	vec3 final = black;
	
	vec3 offset = vec3(push_constants.cursor_position * 2.0 - 1.0, 0.0) * 8.0;
	vec3 camera_position = vec3(0.0, -3.0, -8.5) + offset;
	
	
	const float vertical_fov = 45.0;
	const float aperture = 0.5;
	const float lens_radius = aperture / 2.0;
	const float theta = vertical_fov * PI180;
	const float half_height = tan(theta * 0.5);
	const float half_width = aspect_ratio * half_height;

	mat3 look_at = lookat(camera_position, origin);

	float dist_to_focus = push_constants.cursor_position.x * 5.0 + 5.0;
	dist_to_focus = length(camera_position);

	vec3 lower_left_corner = camera_position - look_at * vec3(half_width, half_height, 1.0) * dist_to_focus;
	vec3 horizontal = 2.0 * half_width * dist_to_focus * look_at[0];
	vec3 vertical = 2.0 * half_height * dist_to_focus * look_at[1];
	
	
	scene_light.area = total_area(scene_light);
	
	vec3 edge0 = scene_light.ur - scene_light.ul;
	vec3 edge1 = scene_light.ll - scene_light.ul;
	scene_light.normal = 
	normalize
	(cross(edge0, edge1));
	
	scene_light.dot_edge0 = dot(edge0, edge0);
	scene_light.dot_edge1 = dot(edge1, edge1);
	

	for (uint j = 0; j < number_of_iterations; ++j)
	{
		// By jittering the uv-coordinates a tiny bit here, we get 
		// "free" anti-aliasing.
		vec2 jitter = { gpu_rnd(seed), gpu_rnd(seed) };
		jitter = jitter * 2.0 - 1.0;
		uv += (jitter / push_constants.resolution) * anti_aliasing;
		
		vec3 rd = lens_radius * vec3(random_on_disk(seed), 0.0);
		vec3 lens_offset = look_at * vec3(rd.xy, 0.0);
		vec3 ro = camera_position + lens_offset;
		rd = lower_left_corner + uv.x * horizontal + uv.y * vertical - camera_position - lens_offset;
		
		rd = normalize(rd);
		
		
		/*
		vec3 ro = camera_position;
		vec3 rd = normalize(lookat(ro, origin) * vec3(uv, 1.0));
		*/

		// Calculate the ray direction based on the current fragment's
		// uv-coordinates. All rays will originate from the camera's
		// location. 		
		ray r = { ro, rd };

		// Define some colors.
		const vec3 sky = black;
		vec3 color = black;
		vec3 accumulated = white;

		// This is the main path tracing loop.
		for (uint i = 0; i < number_of_bounces; ++i)
		{	
			intersection inter = intersect_scene(r);

			//float pct = (gl_FragCoord.x / push_constants.resolution.x);
			//bool next_event_estimation = !(bool(step(0.5, pct)));
			//bool next_event_estimation = false;
			bool next_event_estimation = true;

			if (inter.object_type == object_type_miss)
			{
				color += accumulated * sky;
				break;
			}
			else if (inter.object_type == object_type_area_light)
			{
				if (next_event_estimation && i == 0) 
				{
					// Next event estimation - do nothing here!
					// ...unless it's the first bounce.
					color += scene_light.intensity * accumulated;
				}
				else {
					color += scene_light.intensity * accumulated;
				}
				break;
			}

			// If we get here, that means we've hit an object.
			// ...

			// Calculate the origin and direction of the new, scattered ray.
			material mtl = materials[inter.material_index];
			r.origin = inter.position + inter.normal * epsilon;
			
			
			
			//r.direction = scatter(mtl, inter, seed_a, seed_b);
			

			// Accumulate color.
			if (mtl.type == material_type_diffuse
			// && cos_theta > 0.0
			)	
			{
				
				// Generate a pair of per-bounce random seeds.
				float seed_a = gpu_rnd(seed);
				float seed_b = gpu_rnd(seed);
				
				r.direction = (cos_weighted_hemisphere(inter.normal, seed_a, seed_b));
				
				float cos_theta = 
				max(0.0, 
				dot(inter.normal, r.direction)
				)
				;
			
				//if(cos_theta > 0.0)
				{
					vec3 brdf = mtl.reflectance * PI1;
					accumulated *= PI2 * brdf * cos_theta;

					if (next_event_estimation)
					{	
						color += 
						accumulated * 
						sample_light_source(inter.position, inter.normal, brdf, seed);
					}
				}
			}
			else if (mtl.type == material_type_metallic)
			{
				accumulated *= mtl.reflectance;
				r.direction =  reflect(inter.incident, inter.normal);
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
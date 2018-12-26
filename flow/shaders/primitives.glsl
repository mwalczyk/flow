const int object_type_miss = -1;
const int object_type_sphere = 0;
const int object_type_plane = 1;
const int object_type_box = 2;
const int object_type_quad = 3; 

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
	vec3 bounds[2];
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

quad build_quad(float w, float h, in vec3 center, in vec3 axis, float ang, int material_index)
{	
	float half_w = w * 0.5f;
	float half_h = h * 0.5f;

	vec3 ul = { -half_w,  -half_h, 0.0f }; 
	vec3 ur = {  half_w, -half_h, 0.0f };
	vec3 lr = {  half_w, half_h, 0.0f }; 
	vec3 ll = { -half_w, half_h, 0.0f };

	mat4 rot = rotation(axis, ang);
	ul = (rot * vec4(ul, 1.0f)).xyz + center;
	ur = (rot * vec4(ur, 1.0f)).xyz + center;
	lr = (rot * vec4(lr, 1.0f)).xyz + center;
	ll = (rot * vec4(ll, 1.0f)).xyz + center;

	return quad(ul, ur, lr, ll, material_index);
}

box build_box(in vec3 size, in vec3 center, int material_index)
{
	// Note that positive y is actually down.
	const vec3 b0 = vec3(-0.5f,  0.5f, -0.5f) * size + center; // Min (LB)
	const vec3 b1 = vec3( 0.5f, -0.5f,  0.5f) * size + center; // Max (RT)

	return box(vec3[2](b0, b1), material_index);
}
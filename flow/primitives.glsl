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
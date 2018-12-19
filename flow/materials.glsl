const int material_type_invalid = -1;
const int material_type_diffuse = 0;
const int material_type_metallic = 1;
const int material_type_dielectric = 2;

// A material has three attributes:
//
// 1. The primary color (albedo)
// 2. An integer denoting the type of the material (diffuse, metallic, etc.)
// 3. A flag indicating whether or not this material belongs to a light source
struct material 
{
	vec3 reflectance;
	int type;
	bool is_light;
};

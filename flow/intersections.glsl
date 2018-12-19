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

// An intersection holds several values:
// 1. The material index of the object that was hit (`material_index`)
// 2. The type of object that was hit (or a miss)
// 3. The direction vector of the `incident` ray
// 4. The location of intersection in world space (`position`)
// 5. The `normal` vector of the object that was hit, calculated at `position`
// 6. A scalar value `t`, which denotes a point along the incident ray
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

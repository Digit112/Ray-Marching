#ifndef RAY_MARCHING
#define RAY_MARCHING

#include <math.h>
#include <stdio.h>

#ifndef GPU_ENABLED
	#include <unistd.h>
#endif

#ifdef GPU_ENABLED
	#include "utility_GPU/vec.hpp"
	#include "utility_GPU/color.hpp"
#else
	#include "utility/vec.hpp"
	#include "utility/color.hpp"
#endif

#define COLLISION_DIST 3.5E-4
#define NORMAL_DIST 1E-5
#define MAX_RAY_DIS 50

// Define some macros because functions are different depending on the target device.

#ifdef GPU_ENABLED
	#define DEV_MIN fminf
	#define DEV_MAX fmaxf
#else
	#define DEV_MIN fmin
	#define DEV_MAX fmax
#endif

class object;
class light;
class distance_to_object;
class collision;
class scene_descriptor;
class camera;

using namespace util;

// Function that calculates the distance of a point to the surface of some object.
using dist_f = double (*)(vecd3, float*);

// Function that calculates the color of a pixel given the scene descriptor, camera, and a ray collision.
using shde_f = rgb (*)(const scene_descriptor&, const camera&, const collision&);

class object {
public:
	// The distance function. Note that it receives the distance in local coordinates.
	dist_f distance;
	shde_f shader;
	
	// Used to convert the actual position of the ray to local position BEFORE it gets passed to the distance function.
	vecd3 pos;
	quaternion rot;
	
	float shininess;
	
	rgb diff_color;
	rgb spec_color;
	
	// Index in the params list of the first parameter for this object. The params list is maintained by the scene descriptor.
	int params_ind;
	
	object(); // Does nothing.
	object(dist_f distance, shde_f shader, vecd3 pos, quaternion rot, rgb diff_color, rgb spec_color, float shininess, int params_ind);
};

class light {
public:
	vecd3 pos;
	
	float radius;
	
	float diff_brightness;
	float spec_brightness;
	
	rgb diff_color; // On the bottom of the class for alignment.
	rgb spec_color;
	
	bool is_infinite;
	
	light(); // Does nothing.
	light(vecd3 pos, float radius, float diff_brightness, rgb diff_color, float spec_brightness, rgb spec_color, bool is_infinite);
};

// Used internally to hold the distance function's return.
class distance_to_object {
public:
	double dis; // Distance
	int o_i; // Index of the nearest object.
};

// Used internally to store collision information.
class collision {
public:
	vecd3 pos;
	
	int object_ind;
	
	float approach_min;
	vecd3 approach_pos;
	
	#ifdef GPU_ENABLED
	__device__ __host__
	#endif
	collision();
};

// A list of objects and lights in the scene.
class scene_descriptor {
public:
	// Lists of objects.
	int objects_n;
	object* objects;
	
	// List of lights.
	int lights_n;
	light* lights;
	
	// Parameters for the objects.
	int params_n;
	float* params;
	
	// Ambient light
	float ambient_brightness;
	rgb ambient_color;
	
	scene_descriptor() {}
	
	scene_descriptor(int max_objects_n, int max_params_n, int max_lights_n);
	
	// Sets the amount of ambient lighting. Defaults to 0. Brightness is measured in candela assuming a monitor brightness of 350cd.
	void set_ambience(float brightness, rgb color);
	
	// Adds an object to the scene.
	// Be sure not to overstep the max parameters and max objects set in the constructor!!
	void add_object_with_params(dist_f* distance, shde_f* shader, vecd3 pos, quaternion rot, rgb diff_color, rgb spec_color, float shininess, int params_n, float* params);
	
	// Adds a light to the scene.
	void add_light(vecd3 pos, float radius, float diff_brightness, rgb diff_color, float spec_brightness, rgb spec_color, bool is_infinite);
	
	// Get the distaance to a particular object.
	#ifdef GPU_ENABLED
	__device__
	#endif
	double distance_to_individual(vecd3 pos, int object_ind) const;
	
	// Get the distance to the nearest object and the nearest object.
	#ifdef GPU_ENABLED
	__device__
	#endif
	distance_to_object distance(vecd3 pos) const;
	
	// Raymarch through the scene and the details of the resulting collision.
	#ifdef GPU_ENABLED
	__device__
	#endif
	collision raymarch(vecd3 ray_pos, vecd3 ray_dir) const;
	
	// Raymarch through the scene and the details of the resulting collision.
	#ifdef GPU_ENABLED
	__device__
	#endif
	float AO_raymarch(vecd3 ray_pos, vecd3 ray_dir) const;
	
	// Calculate the raycast forward until adjacent rays would diverge.
	#ifdef GPU_ENABLED
	__device__
	#endif
	collision primary_raymarch(vecd3 ray_pos, vecd3 ray_dir, float delta_dis) const;
	
	// March an individual ray
	#ifdef GPU_ENABLED
	__device__
	#endif
	collision secondary_raymarch(vecd3 ray_pos, vecd3 ray_dir) const;
	
	// Get the normal vector at a point and for a particular object.
	#ifdef GPU_ENABLED
	__device__
	#endif
	vecd3 normal(vecd3 pos, int object_ind) const;
	
	~scene_descriptor();
};

// A camera which can render a scene.
class camera {
public:
	vecd3 pos;
	quaternion rot;
	
	int width;
	int height;
	
	int nsl;
	
	float fov;
	
	uint8_t* img;
	float* accl;
	
	camera(vecd3 pos, quaternion rot, float fov_slope, int width, int height, int nsl);
	
	void render(const scene_descriptor& sd);
	
	~camera();
};

#include "ray_march.cpp"

#endif
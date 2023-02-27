#ifndef RAY_MARCHING
#define RAY_MARCHING

#include <math.h>
#include <stdio.h>
#include <unistd.h>

#include "utility/vec.hpp"
#include "utility/color.hpp"

#define COLLISION_DIST 1E-7
#define NORMAL_DIST 1E-9

class object;
class camera;
class scene_descriptor;

using namespace util;

// Function that calculates the distance of a point to the surface of some object.
using dist_f = double (*)(vecd3, float*);

class object {
public:
	// The distance function. Note that it receives the distance in local coordinates.
	dist_f distance;
	
	// Used to convert the actual position of the ray to local position BEFORE it gets passed to the distance function.
	vecd3 pos;
	quaternion rot;
	
	float shininess;
	
	rgb diff_color;
	rgb spec_color;
	
	// Index in the params list of the first parameter for this object. The params list is maintained by the scene descriptor.
	int params_ind;
	
	object(); // Does nothing.
	object(dist_f distance, vecd3 pos, quaternion rot, rgb diff_color, rgb spec_color, float shininess, int params_ind);
};

class light {
public:
	vecd3 pos;
	
	float diff_brightness;
	float spec_brightness;
	
	rgb diff_color; // On the bottom of the class for alignment.
	rgb spec_color;
	
	light(); // Does nothing.
	light(vecd3 pos, float diff_brightness, rgb diff_color, float spec_brightness, rgb spec_color);
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
	
	scene_descriptor(int max_objects_n, int max_params_n, int max_lights_n);
	
	// Sets the amount of ambient lighting. Defaults to 0. Brightness is measured in candela assuming a monitor brightness of 350cd.
	void set_ambience(float brightness, rgb color);
	
	// Adds an object to the scene.
	// Be sure not to overstep the max parameters and max objects set in the constructor!!
	void add_object_with_params(dist_f distance, vecd3 pos, quaternion rot, rgb diff_color, rgb spec_color, float shininess, int params_n, float* params);
	
	// Adds a light to the scene.
	void add_light(vecd3 pos, float diff_brightness, rgb diff_color, float spec_brightness, rgb spec_color);
	
	// Get the distaance to a particular object.
	double distance_to_individual(vecd3 pos, int object_ind) const;
	
	// Get the distance to the nearest object and the nearest object.
	distance_to_object distance(vecd3 pos) const;
	
	// Raymarch through the scene and the details of the resulting collision.
	collision raymarch(vecd3 ray_pos, vecd3 ray_dir) const;
	
	// Get the normal vector at a point and for a particular object.
	vecd3 normal(vecd3 pos, int object_ind) const;
	
	~scene_descriptor();
};

// A camera which can render a scene.
class camera {
public:
	vecd3 pos;
	quaternion rot;
	
	float fov;
	
	camera(vecd3 pos, quaternion rot, float fov_slope);
	
	void render(const scene_descriptor& sd, int width, int height, uint8_t* img);
};

#include "ray_march.cpp"

#endif
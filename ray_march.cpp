/* ---- primitive distance estimators ---- */

// Params:
//   0 - Radius
#ifdef GPU_ENABLED
__device__
#endif
double SPHERE(vecd3 pos, float* params) {
	return sqrt(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z) - params[0];
}

// Params:
//  0 - front (-x)
//  1 - back (+x)
//  2 - left (-y)
//  3 - right (+y)
//  4 - bottom (-z)
//  5 - top (+z)
#ifdef GPU_ENABLED
__device__
#endif
double CUBE(vecd3 pos, float* params) {
	float dy = pos.y - fmin(fmax(params[2], pos.y), params[3]);
	float dx = pos.x - fmin(fmax(params[0], pos.x), params[1]);
	float dz = pos.z - fmin(fmax(params[4], pos.z), params[5]);
	
	return sqrt(dx*dx + dy*dy + dz*dz);
}

// No Params
#ifdef GPU_ENABLED
__device__
#endif
double INFINITE_PLANE(vecd3 pos, float* params) {
	return pos.z;
}

/* ---- Constants representing the number of parameters each shape uses ---- */

#define SPHERE_N 1
#define INFINITE_PLANE_N 0
#define CUBE_N 6

/* ---- object ---- */

object::object() {}
object::object(dist_f distance, vecd3 pos, quaternion rot, rgb diff_color, rgb spec_color, float shininess, int param_n) :
	distance(distance), pos(pos), rot(rot), diff_color(diff_color), spec_color(spec_color), shininess(shininess), params_ind(params_ind) {}

/* ---- light ---- */

light::light() {}
light::light(vecd3 pos, float diff_brightness, rgb diff_color, float spec_brightness, rgb spec_color) : pos(pos), diff_brightness(diff_brightness), diff_color(diff_color), spec_brightness(spec_brightness), spec_color(spec_color) {}

/* ---- scene_descriptor ---- */

scene_descriptor::scene_descriptor(int max_objects_n, int max_params_n, int max_lights_n) : objects_n(0), params_n(0), lights_n(0), ambient_brightness(0), ambient_color(255, 255, 255) {
	objects = new object[max_objects_n];
	params = new float[max_params_n];
	lights = new light[max_lights_n];
}

void scene_descriptor::set_ambience(float brightness, rgb color) {
	ambient_brightness = brightness;
	ambient_color = color;
}

void scene_descriptor::add_object_with_params(dist_f distance, vecd3 pos, quaternion rot, rgb diff_color, rgb spec_color, float shininess, int p_n, float* p) {
	objects[objects_n].distance = distance;
	objects[objects_n].pos = pos;
	objects[objects_n].rot = rot;
	objects[objects_n].shininess = shininess;
	objects[objects_n].diff_color = diff_color;
	objects[objects_n].spec_color = spec_color;
	
	objects[objects_n].params_ind = params_n;
	objects_n++;
	
	// Copy the provided parameters into the prams array and advance the array length.
	if (p_n > 0) {
		memcpy(params + params_n, p, sizeof(float) * p_n);
		params_n += p_n;
	}
}

void scene_descriptor::add_light(vecd3 pos, float diff_brightness, rgb diff_color, float spec_brightness, rgb spec_color) {
	lights[lights_n].pos = pos;
	lights[lights_n].diff_brightness = diff_brightness;
	lights[lights_n].diff_color = diff_color;
	lights[lights_n].spec_brightness = spec_brightness;
	lights[lights_n].spec_color = spec_color;
	lights_n++;
}

double scene_descriptor::distance_to_individual(vecd3 pos, int object_ind) const {
	// Apply the inverse of the object's transform to the point.
	vecd3 pos_local((~objects[object_ind].rot).apply(pos - objects[object_ind].pos));
	
	// Get the distance from the transformed point to the object.
	return objects[object_ind].distance(pos_local, params + objects[object_ind].params_ind);
}
	

// Gets the minimum distance from a point to all objects specified in the screen descriptor.
distance_to_object scene_descriptor::distance(vecd3 pos) const {
	distance_to_object d = {1E20, -1};
	
	// For each object in the scene, get the distance to the ray pos.
	for (int o = 0; o < objects_n; o++) {
		double dis = distance_to_individual(pos, o);
		
		// Take the min of this distance and min_dis.
		// Record the object with the minimum distance.
		if (dis < d.dis) {
			d.dis = dis;
			d.o_i = o;
		}
	}
	
	return d;
}

// Advances a ray and returns some information about the collision.
collision scene_descriptor::raymarch(vecd3 ray_pos, vecd3 ray_dir) const {
	collision c;
	
	// For each iteration...
	for (int i = 0; i < 400; i++) {
		// Get the distance to the scene.
		distance_to_object d = distance(ray_pos);
		
		// Detect collision.
		if (d.dis < COLLISION_DIST) {
			c.pos = ray_pos;
			c.object_ind = d.o_i;
			return c;
		}
		
		// If the distance is too large, quit.
		if (d.dis > 20) {
			return c;
		}
		
		// Advance the ray "dis" units.
		ray_pos.x += ray_dir.x * d.dis;
		ray_pos.y += ray_dir.y * d.dis;
		ray_pos.z += ray_dir.z * d.dis;
	}
	
	return c;
}

vecd3 scene_descriptor::normal(vecd3 pos, int object_ind) const {
	vecd3 a(
		distance_to_individual(pos + vecd3(NORMAL_DIST, 0, 0), object_ind),
		distance_to_individual(pos + vecd3(0, NORMAL_DIST, 0), object_ind),
		distance_to_individual(pos + vecd3(0, 0, NORMAL_DIST), object_ind)
	);
	
	vecd3 b(
		distance_to_individual(pos - vecd3(NORMAL_DIST, 0, 0), object_ind),
		distance_to_individual(pos - vecd3(0, NORMAL_DIST, 0), object_ind),
		distance_to_individual(pos - vecd3(0, 0, NORMAL_DIST), object_ind)
	);
	
	return (a - b).normalize();
}

scene_descriptor::~scene_descriptor() {
	delete[] objects;
	delete[] params;
	delete[] lights;
}

/* ---- camera ---- */

camera::camera(vecd3 pos, quaternion rot, float fov_slope) : pos(pos), rot(rot), fov(fov_slope) {}

// Renders an image by marching a ray from each pixel.
void camera::render(const scene_descriptor& sd, int width, int height, uint8_t* img) {
	vecd3 ray_pos;
	vecd3 ray_dir;
	for (int y = 0; y < height; y++) {
		if (y % 10 == 0) {
			printf("%d...\n", y);
		}
		
		for (int x = 0; x < width; x++) {
			int ind = ((height - y - 1)*width + x)*3;
			// Set the ray position and direction based on the camera position and orientation, and pixel coordinates.
			ray_pos = pos;
			ray_dir = rot.apply(vecd3(1, ((float) x / width - 0.5) * fov, ((float) y / height - 0.5) * fov)).normalize();
			
				// printf("(%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f)\n", ray_pos.x, ray_pos.y, ray_pos.z, ray_dir.x, ray_dir.y, ray_dir.z);
			
			collision o_col = sd.raymarch(ray_pos, ray_dir);
			
//			printf("(%d, %d): (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f)\n", x, y, ray_pos.x, ray_pos.y, ray_pos.z, ray_dir.x, ray_dir.y, ray_dir.z);
			
//			printf("  %f\n", min_dis);
			
			// Set the pixel color to black if no collision occurred.
			if (o_col.object_ind == -1) {
				img[ind  ] = 0;
				img[ind+1] = 0;
				img[ind+2] = 0;
				continue;
			}
			
			// Get the surface normal at this point.
			vecd3 normal = sd.normal(o_col.pos, o_col.object_ind);
			
			// if (x == 220 && y == 370) {
				// printf("Ray collided at (%.2f, %.2f, %.2f). Normal (%.2f, %.2f, %.2f)\n", o_col.pos.x, o_col.pos.y, o_col.pos.z, normal.x, normal.y, normal.z);
			// }
			
			// Apply Phong reflection
			
			// Add ambient lighting
			float red_brightness = sd.ambient_brightness * sd.ambient_color.r / 255 * sd.objects[o_col.object_ind].diff_color.r / 255;
			float grn_brightness = sd.ambient_brightness * sd.ambient_color.g / 255 * sd.objects[o_col.object_ind].diff_color.g / 255;
			float blu_brightness = sd.ambient_brightness * sd.ambient_color.b / 255 * sd.objects[o_col.object_ind].diff_color.b / 255;
			
			// For each light...
			for (int l = 0; l < sd.lights_n; l++) {
				// Get the dot product between the direction to the light and the normal, for diffuse shading.
				float diff = vecd3::dot((sd.lights[l].pos - o_col.pos).normalize(), normal);
				
				// if (x == 220 && y == 370) {
					// vecd3 diffvec = (sd.lights[l].pos - o_col.pos).normalize();
					// printf("(%.2f, %.2f, %.2f) . (%.2f, %.2f, %.2f) = %.2f\n", diffvec.x, diffvec.y, diffvec.z, normal.x, normal.y, normal.z, diff);
				// }
				
				// Only illuminate if the dot product is positive.
				if (diff >= 0) {
					// Calculate the brightness of this point according to the inverse square law.
					diff *= sd.lights[l].diff_brightness / (o_col.pos - sd.lights[l].pos).sqr_mag();
					
					// if (x == 220 && y == 370) {
						// printf("%.2f\n", diff);
					// }
					
					// Modify the brightness in each color according to the calculated brightness, the light's color, and the object's color.
					red_brightness += diff * sd.lights[l].diff_color.r / 255 * sd.objects[o_col.object_ind].diff_color.r / 255;
					grn_brightness += diff * sd.lights[l].diff_color.g / 255 * sd.objects[o_col.object_ind].diff_color.g / 255;
					blu_brightness += diff * sd.lights[l].diff_color.b / 255 * sd.objects[o_col.object_ind].diff_color.b / 255;
					
					// Get the dot product used for specular highlights.
					float spec = vecd3::dot(vecd3::rev_reflect(o_col.pos - sd.lights[l].pos, normal).normalize(), (pos - o_col.pos).normalize());
					
					// If spec is non-negative, add specular highlights.
					if (spec >= 0) {
						// Raise the dot product to the power of the object's shininess.
						spec = pow(spec, sd.objects[o_col.object_ind].shininess);
						
						// Calculate the brightness of this point according to the inverse square law.
						spec *= sd.lights[l].spec_brightness / (o_col.pos - sd.lights[l].pos).sqr_mag();
						
						// Modify the brightness.
						red_brightness += spec * sd.lights[l].spec_color.r / 255 * sd.objects[o_col.object_ind].spec_color.r / 255;
						grn_brightness += spec * sd.lights[l].spec_color.g / 255 * sd.objects[o_col.object_ind].spec_color.g / 255;
						blu_brightness += spec * sd.lights[l].spec_color.b / 255 * sd.objects[o_col.object_ind].spec_color.b / 255;
					}
				}
			}
			
			// if (x == 220 && y == 370) {
				// printf("Red: %.2f\n", red_brightness);
			// }
			
			// Set the pixel color according to the brightness.
			float final_r = (red_brightness / 350);
			final_r = final_r - log(exp(7 * final_r) + exp(7)) / 7 + 1;
			img[ind] = (uint8_t) (final_r * 255);
			
			float final_g = (grn_brightness / 350);
			final_g = final_g - log(exp(7 * final_g) + exp(7)) / 7 + 1;
			img[ind+1] = (uint8_t) (final_g * 255);
			
			float final_b = (blu_brightness / 350);
			final_b = final_b - log(exp(7 * final_b) + exp(7)) / 7 + 1;
			img[ind+2] = (uint8_t) (final_b * 255);
		}
	}
}

/* ---- collision ---- */

collision::collision() : pos(0, 0, 0), object_ind(-1) {}
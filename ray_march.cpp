bool debug = false;
#define DEBUG_X 197
#define DEBUG_Y 160

/* ---- The Render Function ---- */

// Renders an image by marching a ray from each pixel.
// This cannot be a member function because __global__ functions must not be members of classes, which is actually kind of a bummer.
#ifdef GPU_ENABLED
__global__
#endif
void cast_primary(const scene_descriptor& sd, const camera& cam, int nsl, int width, int height, float* accl) {
	vecd3 ray_pos;
	vecd3 ray_dir;
	
	int real_width = width * (nsl*2 + 1);
	int real_height = height * (nsl*2 + 1);
	
	// Calculate the rate at which primary ray and the most distance secondary ray diverge.
	float delta_dis_x = cam.fov / width;
	float delta_dis_y = cam.fov / height;
	float delta_dis = sqrt(delta_dis_x*delta_dis_x + delta_dis_y*delta_dis_y);
	
	// If compiling for GPU, calculate x and y from GPU thread information.
	#ifdef GPU_ENABLED
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// printf("Rendering (%d, %d)\n", x, y);
	
	// Otherwise, loop over all threads.
	#else
	for (int y = 0; y < height; y++) {
		if (y % 10 == 0) {
			printf("Casting Primary rays for row %d...\n", y);
		}
		
		for (int x = 0; x < width; x++) {
	#endif
			// #ifdef GPU_ENABLED
				// bool debug = false;
			// #else
				// debug = false;
			// #endif
			
			// if (x == DEBUG_X && y == DEBUG_Y) {
				// debug = true;
			// }
			
			// Find the index of this pixel and set its color to black.
			int rev_y = (height - y - 1);
			int real_x = x     * (nsl * 2 + 1) + nsl;
			int real_y = rev_y * (nsl * 2 + 1) + nsl;
			
			#ifdef GPU_ENABLED
//				if (debug) printf("Painting (%d*%d+%d, %d*%d+%d) -> (%d, %d) -> (%d, %d)\n", blockIdx.x, blockDim.x, threadIdx.x, blockIdx.y, blockDim.y, threadIdx.y, x, y, x, rev_y);
			#else
//				if (debug) printf("(%d, %d) -> (%d, %d)\n", x, y, x, rev_y);
			#endif
			
			// Set the ray position and direction based on the camera position and orientation, and pixel coordinates.
			ray_pos = cam.pos;
			ray_dir = cam.rot.apply(vecd3(1, ((float) real_x / real_width - 0.5) * cam.fov, ((float) real_y / real_height - 0.5) * cam.fov)).normalize();
			
//			printf("Colliding...\n");
			
			// March primary rays until secondary rays diverge.
			collision o_col = sd.primary_raymarch(ray_pos, ray_dir, delta_dis);
			
//			if (debug) printf("Primary ray (%f, %f, %f) -> (%f, %f, %f) diverged at (%f, %f, %f)\n", ray_pos.x, ray_pos.y, ray_pos.z, ray_dir.x, ray_dir.y, ray_dir.z, o_col.pos.x, o_col.pos.y, o_col.pos.z);

			// Write the distance to divergence to the acceleration structure.
			// Write -1 if the primary ray escaped before diverging.
			if (o_col.object_ind != -1) {
				accl[y*width+x] = (cam.pos - o_col.pos).mag();
			}
			else {
				accl[y*width+x] = -1;
			}
	
	#ifndef GPU_ENABLED
		}
	}
	#endif
}

#ifdef GPU_ENABLED
__global__
#endif
void cast_secondary(const scene_descriptor& sd, const camera& cam, int nsl, int width, int height, uint8_t* img, float* accl) {
	vecd3 ray_pos;
	vecd3 ray_dir;
	
	int real_width = width * (nsl*2 + 1);
	int real_height = height * (nsl*2 + 1);
	
	// If compiling for GPU, calculate x and y from GPU thread information.
	#ifdef GPU_ENABLED
	int real_x = blockIdx.x * blockDim.x + threadIdx.x;
	int real_y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// printf("Rendering (%d, %d)\n", x, y);
	
	// Otherwise, loop over all threads.
	#else
	for (int real_y = 0; real_y < real_height; real_y++) {
		
		if (real_y % 10 == 0) {
			printf("Casting Secondary rays for row %d...\n", real_y);
		}
		
		for (int real_x = 0; real_x < real_width; real_x++) {
	#endif
			int x = real_x / (nsl * 2 + 1);
			int y = real_y / (nsl * 2 + 1);
			
			float distance_to_divergence = accl[y*width + x];
			
//			printf("(%d, %d)\n", real_x, real_y);
			
			// March secondary rays unless the primary ray escaped.
//				printf("Primary ray did not escape.\n");
			
//				if (debug) printf("Launching secondary rays corresponding to primary ray (%d, %d) -> (%d, %d) %.2f from the camera\n", x, rev_y, real_x, real_y, distance_to_divergence);
			int ind = (real_y * real_width + real_x)*3;
			
//				if (debug) printf("  Painting (%d, %d) -> %d\n", a, b, ind);
			
			img[ind  ] = 0;
			img[ind+1] = 0;
			img[ind+2] = 0;
			
			// TODO: distance to divergence is being reduced to account for an unknown flaw where secondary rays are instantiated slightly too far from the camera.
			// Issue is best replicated with the camera facing straight down at a plane.
			// This issue has been effectively resolved to the best of my knowledge. Fixing it properly would be a small performance improvement.
			vecd3 s_ray_dir = cam.rot.apply(vecd3(1, ((float) real_x / real_width - 0.5) * cam.fov, ((float) (real_width - real_y - 1) / real_height - 0.5) * cam.fov)).normalize();
			vecd3 s_ray_pos = cam.pos + s_ray_dir * (distance_to_divergence * 0 * vecd3::dot(ray_dir, s_ray_dir));
			
//				printf("%f: (%f, %f, %f) -> (%f, %f, %f)\n", distance_to_divergence, s_ray_pos.x, s_ray_pos.y, s_ray_pos.z, s_ray_dir.x, s_ray_dir.y, s_ray_dir.z);
			collision s_col = sd.secondary_raymarch(s_ray_pos, s_ray_dir);
			
//				if (debug) printf("  (%f, %f, %f) -> (%f, %f, %f) collides with %d at (%f, %f, %f)\n", s_ray_pos.x, s_ray_pos.y, s_ray_pos.z, s_ray_dir.x, s_ray_dir.y, s_ray_dir.z, s_col.object_ind, s_col.pos.x, s_col.pos.y, s_col.pos.z);
	
			// Set the pixel color to brightness based on the nearest approach
			// TODO: Allow objects to have custom glows. The code currently generates the glow used in my Mandelbulb render.
			// Swap out the code inside the if-block with just "color = {0, 0, 0};" to remove the glow.
			// NOTE: The glow will be inaccurate due to the fact that the primary rays don't send their closest approaches to the secondary rays that they spawn.
			rgb color;
			if (s_col.object_ind == -1) {
				float glow = 0.15 / ((float) 15 * s_col.approach_min + 1);
				
				float hue = s_col.approach_pos.mag() / 1.2 * 360;
				rgb glow_color(hsv(hue, 0.75, glow));
				
				color = {glow_color.r * 255, glow_color.g * 255, glow_color.b * 255};
			}
			else {
				// Get the color from this object's shader.
				color = sd.objects[s_col.object_ind].shader(sd, cam, s_col);
			}
			
			img[ind  ] = color.r;
			img[ind+1] = color.g;
			img[ind+2] = color.b;
	#ifndef GPU_ENABLED
		}
	}
	#endif
}

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
//	printf("CUBE\n");
	float dy = pos.y - DEV_MIN(DEV_MAX(params[2], pos.y), params[3]);
	float dx = pos.x - DEV_MIN(DEV_MAX(params[0], pos.x), params[1]);
	float dz = pos.z - DEV_MIN(DEV_MAX(params[4], pos.z), params[5]);
	
	return sqrt(dx*dx + dy*dy + dz*dz);
}

// No Params
#ifdef GPU_ENABLED
__device__
#endif
double INFINITE_PLANE(vecd3 pos, float* params) {
//	printf("INFINITE_PLANE\n");
	return pos.z;
}

// No Params
#ifdef GPU_ENABLED
__device__
#endif
double MANDELBULB(vecd3 pos, float* params) {
	float power = params[0];
	
	vecd3 z = pos;
	float dr = 1;
	float r = 0;
	
	for (int i = 0; i < 30; i++) {
		r = sqrt(z.x*z.x + z.y*z.y + z.z*z.z);
		
		if (r > MAX_RAY_DIS) {
			break;
		}
		
		// Convert to polar coordinates.
		float theta = acos(z.z / r);
		float phi = atan2(z.y, z.x);
		dr = pow(r, power-1)*power*dr + 1;
		
		// Scale and rotate the point.
		float zr = pow(r, power);
		theta *= power;
		phi *= power;
		
		// Convert back to cartesian coordinates
		z = vecd3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta)) * zr;
		
		z = z + pos;
	}
	
	return 0.5 * log(r) * r / dr;
}

/* ---- Constants representing the number of parameters each shape uses ---- */

#define SPHERE_N 1
#define INFINITE_PLANE_N 0
#define CUBE_N 6
#define MANDELBULB_N 1

/* ---- Default shaders ---- */

// Solid color.
template <uint8_t r, uint8_t g, uint8_t b>
#ifdef GPU_ENABLED
__device__
#endif
rgb SOLID(const scene_descriptor& sd, const camera& c, const collision& o_col) {
	return rgb(r, g, b);
}

// High-quality diffuse and specular shading.
#ifdef GPU_ENABLED
__device__
#endif
rgb PHONG(const scene_descriptor& sd, const camera& c, const collision& o_col) {
//	printf("Distance a: %x\n", sd.objects[0].distance);
	
	// Get the surface normal at this point.
	vecd3 normal = sd.normal(o_col.pos, o_col.object_ind);
			
//	if (debug) printf("Normal (%.2f, %.2f, %.2f)\n", normal.x, normal.y, normal.z);
	
	// Add ambient lighting
	float red_brightness = sd.ambient_brightness * sd.ambient_color.r / 255 * sd.objects[o_col.object_ind].diff_color.r / 255;
	float grn_brightness = sd.ambient_brightness * sd.ambient_color.g / 255 * sd.objects[o_col.object_ind].diff_color.g / 255;
	float blu_brightness = sd.ambient_brightness * sd.ambient_color.b / 255 * sd.objects[o_col.object_ind].diff_color.b / 255;

	// For each light...
	for (int l = 0; l < sd.lights_n; l++) {
		// Get the direction from the collision to the light, normalized, and the distance and squared distance.
		vecd3 light_dir;
		if (sd.lights[l].is_infinite) {
			light_dir = sd.lights[l].pos;
		}
		else {
			light_dir = sd.lights[l].pos - o_col.pos;
		}
		
		float light_sqr_dis = light_dir.sqr_mag();
		float light_dis = sqrt(light_sqr_dis);
		
		light_dir = light_dir / light_dis;
		
//		if (debug) printf(" Considering light %d in direction %.2f * (%.2f, %.2f, %.2f)\n", l, light_dis, light_dir.x, light_dir.y, light_dir.z);
		
		/* ---- Shadow ---- */
		
		// Get whether this point is in shadow.
		collision l_col;
		if (sd.lights[l].is_infinite) {
			vecd3 light_position = o_col.pos + light_dir * (MAX_RAY_DIS * 0.9);
			l_col = sd.raymarch(light_position, -light_dir);
//			if (debug) printf("  Light is infinite;     Considering light ray (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f)\n", light_position.x, light_position.y, light_position.z, -light_dir.x, -light_dir.y, -light_dir.z);
		}
		else {
			l_col = sd.raymarch(sd.lights[l].pos, -light_dir);
//			if (debug) printf("  Light is not infinite; Considering light ray (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f)\n", sd.lights[l].pos.x, sd.lights[l].pos.y, sd.lights[l].pos.z, -light_dir.x, -light_dir.y, -light_dir.z);
		}
		
//		if (debug) printf("  Light ray collided with %d at (%.2f, %.2f, %.2f).\n  Its closest approach was %.2f at (%.2f, %.2f, %.2f)\n", l_col.object_ind, l_col.pos.x, l_col.pos.y, l_col.pos.z, l_col.approach_min, l_col.approach_pos.x, l_col.approach_pos.y, l_col.approach_pos.z);
		
		// Calculated the darkness of the shadow.
		float shade = 0;
		if (l_col.object_ind == -1 || (l_col.pos - o_col.pos).mag() < COLLISION_DIST * 20) {
			// Infinite lights cast sharp shadows.
			if (sd.lights[l].is_infinite) {
				shade = 1;
			}
			else {
				// Get the light's size in radians
				float light_view = sd.lights[l].radius / (3.14159 * light_dis);
				
				// Get the closest approach's size in radians
				float aprch_view = l_col.approach_min / (3.14159 * (o_col.pos - l_col.approach_pos).mag());
				
				// If some of the light is occluded...
				if (aprch_view < light_view) {
					shade = aprch_view / light_view;
				}
				else {
					shade = 1;
				}
			}
		}
		
//		if (debug) printf("  Shade: %.2f\n", shade);

		if (shade > 0) {
			/* ---- Diffuse reflection ---- */
			
			// Get the dot product between the direction to the light and the normal, for diffuse shading.
			float diff = vecd3::dot(light_dir, normal);
			
			// if (x == 220 && y == 370) {
				// vecd3 diffvec = (sd.lights[l].pos - o_col.pos).normalize();
				// printf("(%.2f, %.2f, %.2f) . (%.2f, %.2f, %.2f) = %.2f\n", diffvec.x, diffvec.y, diffvec.z, normal.x, normal.y, normal.z, diff);
			// }
			
			// Only illuminate if the dot product is positive.
			if (diff >= 0) {
				// Calculate the brightness of this point according to the inverse square law.
				diff *= sd.lights[l].diff_brightness;
				if (!sd.lights[l].is_infinite) {
					diff /= light_sqr_dis;
				}
				
				// if (x == 220 && y == 370) {
					// printf("%.2f\n", diff);
				// }
				
				// Modify the brightness in each color according to the calculated brightness, the light's color, and the object's color.
				red_brightness += shade * diff * sd.lights[l].diff_color.r / 255 * sd.objects[o_col.object_ind].diff_color.r / 255;
				grn_brightness += shade * diff * sd.lights[l].diff_color.g / 255 * sd.objects[o_col.object_ind].diff_color.g / 255;
				blu_brightness += shade * diff * sd.lights[l].diff_color.b / 255 * sd.objects[o_col.object_ind].diff_color.b / 255;
				
				/* ---- Specular reflection ---- */
				
				// Get the direction from the collision to the camera's position.
				vecd3 viewer_dir = (c.pos - o_col.pos).normalize();
				
				// Get the direction a perfectly reflected ray would be traveling.
				vecd3 reflected_dir = vecd3::reflect(light_dir, normal);
				
				// Get the dot product used for specular highlights.
				float spec = vecd3::dot(reflected_dir, viewer_dir);
				
				// If spec is non-negative, add specular highlights.
				if (spec >= 0) {
					// Raise the dot product to the power of the object's shininess.
					spec = pow(spec, sd.objects[o_col.object_ind].shininess);
					
					// Calculate the brightness of this point according to the inverse square law.
					spec *= sd.lights[l].spec_brightness;
					if (!sd.lights[l].is_infinite) {
						spec /= light_sqr_dis;
					}
					
					// Modify the brightness.
					red_brightness += shade * spec * sd.lights[l].spec_color.r / 255 * sd.objects[o_col.object_ind].spec_color.r / 255;
					grn_brightness += shade * spec * sd.lights[l].spec_color.g / 255 * sd.objects[o_col.object_ind].spec_color.g / 255;
					blu_brightness += shade * spec * sd.lights[l].spec_color.b / 255 * sd.objects[o_col.object_ind].spec_color.b / 255;
				}
			}
		}
	}

	// Set the pixel color according to the brightness.
	float final_r = (red_brightness / 350);
	final_r = final_r - log(exp(7 * final_r) + exp(7.0)) / 7 + 1;

	float final_g = (grn_brightness / 350);
	final_g = final_g - log(exp(7 * final_g) + exp(7.0)) / 7 + 1;

	float final_b = (blu_brightness / 350);
	final_b = final_b - log(exp(7 * final_b) + exp(7.0)) / 7 + 1;
	
	return rgb((uint8_t) (final_r * 255), (uint8_t) (final_g * 255), (uint8_t) (final_b * 255));
}

#ifdef GPU_ENABLED
__device__
#endif
rgb MANDELBULB_SHADER(const scene_descriptor& sd, const camera& c, const collision& o_col) {
	// #ifdef GPU_ENABLED
		// int x = blockIdx.x * blockDim.x + threadIdx.x;
		// int y = blockIdx.y * blockDim.y + threadIdx.y;
		// int debug = false;
		// if (x == DEBUG_X && y == DEBUG_Y) {
			// debug = true;
		// }
	// #endif
	
	// Get the surface normal at this point.
	vecd3 normal = sd.normal(o_col.pos, o_col.object_ind);
			
//	if (debug) printf("Normal (%.2f, %.2f, %.2f)\n", normal.x, normal.y, normal.z);
	
	// Add ambient lighting
	float red_brightness = sd.ambient_brightness * sd.ambient_color.r / 255 * sd.objects[o_col.object_ind].diff_color.r / 255;
	float grn_brightness = sd.ambient_brightness * sd.ambient_color.g / 255 * sd.objects[o_col.object_ind].diff_color.g / 255;
	float blu_brightness = sd.ambient_brightness * sd.ambient_color.b / 255 * sd.objects[o_col.object_ind].diff_color.b / 255;

	// For each light...
	for (int l = 0; l < sd.lights_n; l++) {
		// Get the direction from the collision to the light, normalized, and the distance and squared distance.
		vecd3 light_dir;
		if (sd.lights[l].is_infinite) {
			light_dir = sd.lights[l].pos;
		}
		else {
			light_dir = sd.lights[l].pos - o_col.pos;
		}
		
		float light_sqr_dis = light_dir.sqr_mag();
		float light_dis = sqrt(light_sqr_dis);
		
		light_dir = light_dir / light_dis;
		
//		if (debug) printf(" Considering light %d in direction %.2f * (%.2f, %.2f, %.2f)\n", l, light_dis, light_dir.x, light_dir.y, light_dir.z);
		
		/* ---- Shadow ---- */
		
		// Get whether this point is in shadow.
		collision l_col;
		if (sd.lights[l].is_infinite) {
			vecd3 light_position = o_col.pos + light_dir * (MAX_RAY_DIS * 0.9);
			l_col = sd.raymarch(light_position, -light_dir);
//			if (debug) printf("  Light is infinite;     Considering light ray (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f)\n", light_position.x, light_position.y, light_position.z, -light_dir.x, -light_dir.y, -light_dir.z);
		}
		else {
			l_col = sd.raymarch(sd.lights[l].pos, -light_dir);
//			if (debug) printf("  Light is not infinite; Considering light ray (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f)\n", sd.lights[l].pos.x, sd.lights[l].pos.y, sd.lights[l].pos.z, -light_dir.x, -light_dir.y, -light_dir.z);
		}
		
//		if (debug) printf("  Light ray collided with %d at (%.2f, %.2f, %.2f).\n  Its closest approach was %.2f at (%.2f, %.2f, %.2f)\n", l_col.object_ind, l_col.pos.x, l_col.pos.y, l_col.pos.z, l_col.approach_min, l_col.approach_pos.x, l_col.approach_pos.y, l_col.approach_pos.z);
		
		// Calculated the darkness of the shadow.
		float shade = 0;
		if (l_col.object_ind == -1 || (l_col.pos - o_col.pos).mag() < COLLISION_DIST * 20) {
			// Infinite lights cast sharp shadows.
			if (sd.lights[l].is_infinite) {
				shade = 1;
			}
			else {
				// Get the light's size in radians
				float light_view = sd.lights[l].radius / (3.14159 * light_dis);
				
				// Get the closest approach's size in radians
				float aprch_view = l_col.approach_min / (3.14159 * (o_col.pos - l_col.approach_pos).mag());
				
				// If some of the light is occluded...
				if (aprch_view < light_view) {
					shade = aprch_view / light_view;
				}
				else {
					shade = 1;
				}
			}
		}
		
//		if (debug) printf("  Shade: %.2f\n", shade);

		if (shade > 0) {
			/* ---- Diffuse reflection ---- */
			
			// Get the dot product between the direction to the light and the normal, for diffuse shading.
			float diff = vecd3::dot(light_dir, normal);
			
//			if (debug) printf("  (%.2f, %.2f, %.2f) . (%.2f, %.2f, %.2f) = %.2f\n", light_dir.x, light_dir.y, light_dir.z, normal.x, normal.y, normal.z, diff);
			
			// Only illuminate if the dot product is positive.
			if (diff >= 0) {
				// Calculate the brightness of this point according to the inverse square law.
				diff *= sd.lights[l].diff_brightness;
				if (!sd.lights[l].is_infinite) {
					diff /= light_sqr_dis;
				}
				
//				if (debug) printf("  %.2f\n", diff);
				
//				if (debug) printf("  Adding color (%.2f, %.2f, %.2f)\n",
//					shade * diff * sd.lights[l].diff_color.r / 255 * sd.objects[o_col.object_ind].diff_color.r / 255,
//					shade * diff * sd.lights[l].diff_color.g / 255 * sd.objects[o_col.object_ind].diff_color.g / 255,
//					shade * diff * sd.lights[l].diff_color.b / 255 * sd.objects[o_col.object_ind].diff_color.b / 255);
				
				// Modify the brightness in each color according to the calculated brightness, the light's color, and the object's color.
				red_brightness += shade * diff * sd.lights[l].diff_color.r / 255 * sd.objects[o_col.object_ind].diff_color.r / 255;
				grn_brightness += shade * diff * sd.lights[l].diff_color.g / 255 * sd.objects[o_col.object_ind].diff_color.g / 255;
				blu_brightness += shade * diff * sd.lights[l].diff_color.b / 255 * sd.objects[o_col.object_ind].diff_color.b / 255;
			}
		}
	}
	
	// Apply ambient occlusion
	float AO = sd.AO_raymarch(o_col.pos, normal);

	// Set the pixel color according to the brightness.
	float final_r = (red_brightness * AO / 350);
	final_r = final_r - log(exp(7 * final_r) + exp(7.0)) / 7 + 1;

	float final_g = (grn_brightness * AO / 350);
	final_g = final_g - log(exp(7 * final_g) + exp(7.0)) / 7 + 1;

	float final_b = (blu_brightness * AO / 350);
	final_b = final_b - log(exp(7 * final_b) + exp(7.0)) / 7 + 1;
	
	float hue = o_col.pos.mag() / 1.2 * 360;
	rgb point_color(hsv(hue, 0.75, 1));
	
//	if (debug) printf("Final Color: (%.2f, %.2f, %.2f)\n", final_r, final_g, final_b);
	
	return rgb((uint8_t) (final_r * point_color.r * 255), (uint8_t) (final_g * point_color.g * 255), (uint8_t) (final_b * point_color.b * 255));
}

/* ---- Device pointers to functions ---- */

// Yes, this is necessary.
#ifdef GPU_ENABLED
	__device__ dist_f SPHERE_GPU = SPHERE;
	__device__ dist_f INFINITE_PLANE_GPU = INFINITE_PLANE;
	__device__ dist_f CUBE_GPU = CUBE;
	__device__ dist_f MANDELBULB_GPU = MANDELBULB;

	__device__ shde_f PHONG_GPU = PHONG;
	__device__ shde_f MANDELBULB_SHADER_GPU = MANDELBULB_SHADER;
#else
	dist_f SPHERE_CPU = SPHERE;
	dist_f INFINITE_PLANE_CPU = INFINITE_PLANE;
	dist_f CUBE_CPU = CUBE;
	dist_f MANDELBULB_CPU = MANDELBULB;

	shde_f PHONG_CPU = PHONG;
	shde_f MANDELBULB_SHADER_CPU = MANDELBULB_SHADER;
#endif

/* ---- object ---- */

object::object() {}
object::object(dist_f distance, shde_f shader, vecd3 pos, quaternion rot, rgb diff_color, rgb spec_color, float shininess, int param_n) :
	distance(distance), shader(shader), pos(pos), rot(rot), diff_color(diff_color), spec_color(spec_color), shininess(shininess), params_ind(params_ind) {}

/* ---- light ---- */

light::light() {}
light::light(vecd3 pos, float radius, float diff_brightness, rgb diff_color, float spec_brightness, rgb spec_color, bool is_infinite) :
	pos(pos), radius(radius), diff_brightness(diff_brightness), diff_color(diff_color), spec_brightness(spec_brightness), spec_color(spec_color), is_infinite(is_infinite) {}

/* ---- collision ---- */

collision::collision() : pos(0, 0, 0), object_ind(-1), approach_min(1E20), approach_pos(0, 0, 0) {}

/* ---- scene_descriptor ---- */

scene_descriptor::scene_descriptor(int max_objects_n, int max_params_n, int max_lights_n) : objects_n(0), params_n(0), lights_n(0), ambient_brightness(0), ambient_color(255, 255, 255) {
	#ifdef GPU_ENABLED
		cudaError_t err;
		
		err = cudaMallocManaged(&objects, max_objects_n * sizeof(object));
		if (err != cudaSuccess) {
			printf("Allocated Objects: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		err = cudaMallocManaged(&params, max_params_n * sizeof(float));
		if (err != cudaSuccess) {
			printf("Allocated Params: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		err = cudaMallocManaged(&lights, max_lights_n * sizeof(light));
		if (err != cudaSuccess) {
			printf("Allocated Lights: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
	#else
		objects = new object[max_objects_n];
		params = new float[max_params_n];
		lights = new light[max_lights_n];
	#endif
}

void scene_descriptor::set_ambience(float brightness, rgb color) {
	ambient_brightness = brightness;
	ambient_color = color;
}

// Set function pointers.
void scene_descriptor::add_object_with_params(dist_f* distance, shde_f* shader, vecd3 pos, quaternion rot, rgb diff_color, rgb spec_color, float shininess, int p_n, float* p) {
#ifdef GPU_ENABLED
	cudaError_t err;
	
	err = cudaMemcpyFromSymbol(&(objects[objects_n].distance), *distance, sizeof(dist_f));
	if (err != cudaSuccess) {
		printf("Copying Distance function: %s\n", cudaGetErrorName(err));
		exit(1);
	}
	
	err = cudaMemcpyFromSymbol(&(objects[objects_n].shader), *shader, sizeof(shde_f));
	if (err != cudaSuccess) {
		printf("Copying Distance function: %s\n", cudaGetErrorName(err));
		exit(1);
	}
#else
	objects[objects_n].distance = *distance;
	objects[objects_n].shader = *shader;
#endif
	
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

void scene_descriptor::add_light(vecd3 pos, float radius, float diff_brightness, rgb diff_color, float spec_brightness, rgb spec_color, bool is_infinite) {
	if (is_infinite) {
		lights[lights_n].pos = pos.normalize();
	}
	else {
		lights[lights_n].pos = pos;
	}
	
	lights[lights_n].radius = radius;
	lights[lights_n].diff_brightness = diff_brightness;
	lights[lights_n].diff_color = diff_color;
	lights[lights_n].spec_brightness = spec_brightness;
	lights[lights_n].spec_color = spec_color;
	lights[lights_n].is_infinite = is_infinite;
	lights_n++;
}

#ifdef GPU_ENABLED
__device__
#endif
double scene_descriptor::distance_to_individual(vecd3 pos, int object_ind) const {
//	printf("Finding distance from (%.2f, %.2f, %.2f) to object %d\n", pos.x, pos.y, pos.z, object_ind);
	
	// Apply the inverse of the object's transform to the point.
	vecd3 pos_local((~objects[object_ind].rot).apply(pos - objects[object_ind].pos));
	
//	printf("Local position is (%.2f, %.2f, %.2f)\n", pos_local.x, pos_local.y, pos_local.z);
	
	// Get the distance from the transformed point to the object.
	return objects[object_ind].distance(pos_local, params + objects[object_ind].params_ind);
}
	

// Gets the minimum distance from a point to all objects specified in the screen descriptor.
#ifdef GPU_ENABLED
__device__
#endif
distance_to_object scene_descriptor::distance(vecd3 pos) const {
	distance_to_object d = {1E20, -1};
	
	// For each object in the scene, get the distance to the ray pos.
	for (int o = 0; o < objects_n; o++) {
		double dis = distance_to_individual(pos, o);
		
//		printf("Done.\n");
		
		// Take the min of this distance and min_dis.
		// Record the object with the minimum distance.
		if (dis < d.dis) {
			d.dis = dis;
			d.o_i = o;
		}
	}
	return d;
}

#ifdef GPU_ENABLED
__device__
#endif
collision scene_descriptor::raymarch(vecd3 ray_pos, vecd3 ray_dir) const {
	collision c;
	
	vecd3 approach_pos_t;
	float approach_min_t = 1E20;
	
//	if (debug) printf("   Ray Marching (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f)\n", ray_pos.x, ray_pos.y, ray_pos.z, ray_dir.x, ray_dir.y, ray_dir.z);
	
	// For each iteration...
	for (int i = 0; i < 3200; i++) {
		// Get the distance to the scene.
		distance_to_object d = distance(ray_pos);
		
//		if (debug) printf("    %d: |(%.2f, %.2f, %.2f)| = %.2f\n", i, ray_pos.x, ray_pos.y, ray_pos.z, d.dis);
		
		// Detect collision.
		if (d.dis < COLLISION_DIST) {
			c.pos = ray_pos;
			c.object_ind = d.o_i;
//			if (debug) printf("    Collided.\n");
			return c;
		}
		
		// If the distance is too large, quit.
		if (d.dis > MAX_RAY_DIS) {
//			if (debug) printf("    Escaped.\n");
			return c;
		}
		
		// Record the closest approach.
		if (d.dis < approach_min_t) {
			approach_min_t = d.dis;
			approach_pos_t = ray_pos;
		}
		else if (d.dis > approach_min_t * 2) {
			c.approach_min = approach_min_t;
			c.approach_pos = approach_pos_t;
		}
		
//		if (debug) printf("Advancing %.2f * (%.2f, %.2f, %.2f)\n", d.dis, ray_dir.x, ray_dir.y, ray_dir.z);
		
		// Advance the ray "dis" units.
		ray_pos.x += ray_dir.x * d.dis;
		ray_pos.y += ray_dir.y * d.dis;
		ray_pos.z += ray_dir.z * d.dis;
	}
	
	return c;
}

#ifdef GPU_ENABLED
__device__
#endif
float scene_descriptor::AO_raymarch(vecd3 ray_pos, vecd3 ray_dir) const {
//	if (debug) printf("   Ray Marching (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f)\n", ray_pos.x, ray_pos.y, ray_pos.z, ray_dir.x, ray_dir.y, ray_dir.z);

	float prev_dis = 0;
	float initial_dis = distance(ray_pos).dis;
	
	// For each iteration...
	int i;
	for (i = 0; i < 5; i++) {
		// Get the distance to the scene.
		distance_to_object d = distance(ray_pos);
		
//		if (debug) printf("    %d: |(%.2f, %.2f, %.2f)| = %.2f\n", i, ray_pos.x, ray_pos.y, ray_pos.z, d.dis);
		
		// If the distance is too large, quit.
		if (d.dis > MAX_RAY_DIS) {
//			if (debug) printf("    Escaped.\n");
			break;
		}
		
		// Break once we get a distance less than the previous one.
		if (d.dis < prev_dis) {
			break;
		}

		prev_dis = d.dis;
		
//		if (debug) printf("Advancing %.2f * (%.2f, %.2f, %.2f)\n", d.dis, ray_dir.x, ray_dir.y, ray_dir.z);
		
		// Advance the ray "dis" units.
		ray_pos.x += ray_dir.x * d.dis;
		ray_pos.y += ray_dir.y * d.dis;
		ray_pos.z += ray_dir.z * d.dis;
	}
	
	return DEV_MIN(DEV_MAX(log2(prev_dis / initial_dis) / i, 0), 0.9) * 4 / 9 + 0.6;
}

#ifdef GPU_ENABLED
__device__
#endif
collision scene_descriptor::primary_raymarch(vecd3 ray_pos, vecd3 ray_dir, float delta_dis) const {
	#ifdef GPU_ENABLED
		// int x = blockIdx.x * blockDim.x + threadIdx.x;
		// int y = blockIdx.y * blockDim.y + threadIdx.y;
		// bool debug = false;
		// if (x == DEBUG_X && y == DEBUG_Y) {
			// debug = true;
		// }
	#endif
	
	collision c;
	
	vecd3 initial_pos = ray_pos;
	vecd3 previous_pos = ray_pos;
	
//	if (debug) printf("   Ray Marching (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f)\n", ray_pos.x, ray_pos.y, ray_pos.z, ray_dir.x, ray_dir.y, ray_dir.z);
	distance_to_object d = distance(ray_pos);
	
	// For each iteration...
	for (int i = 0; i < 6400; i++) {
		float r1 = d.dis;
		
		// Get the distance to the scene.
		d = distance(ray_pos);
		
//		if (debug) printf("    %d: |(%.2f, %.2f, %.2f)| = %.2f\n", i, ray_pos.x, ray_pos.y, ray_pos.z, d.dis);
		
		// Detect collision.
		if (d.dis < COLLISION_DIST) {
			c.pos = ray_pos;
			c.object_ind = d.o_i;
//			if (debug) printf("    Collided.\n");
			return c;
		}
		
		// If the distance is too large, quit.
		if (d.dis > MAX_RAY_DIS) {
//			if (debug) printf("    Escaped.\n");
			return c;
		}
		
//		if (debug) printf("Advancing %.2f * (%.2f, %.2f, %.2f)\n", d.dis, ray_dir.x, ray_dir.y, ray_dir.z);

		previous_pos = ray_pos;
		
		// Advance the ray "dis" units.
		ray_pos.x += ray_dir.x * d.dis;
		ray_pos.y += ray_dir.y * d.dis;
		ray_pos.z += ray_dir.z * d.dis;
		
		// Get the maximum distance that a secondary ray is allowed to diverge from this ray.
		float max_dis_temp = 1 - (d.dis * d.dis) / (2 * r1 * r1);
		float max_dis = r1 * sqrt(1 - max_dis_temp*max_dis_temp);
		
//		if (debug) printf("  %d: (%.2f, %.2f, %.2f) %f: (%f, %f) -> %f < %f\n", i, ray_pos.x, ray_pos.y, ray_pos.z, delta_dis, r1, d.dis, max_dis, delta_dis * (ray_pos - initial_pos).mag());
		
		// If the max_dis is less than the divergence of the farthest secondary ray, return. Secondary rays will be launched immediately after by render()
		if (max_dis < delta_dis * (ray_pos - initial_pos).mag()) {
			c.pos = previous_pos;
			c.object_ind = d.o_i;
			return c;
		}
	}
	
	return c;
}

// Advances a ray and returns some information about the collision.
#ifdef GPU_ENABLED
__device__
#endif
collision scene_descriptor::secondary_raymarch(vecd3 ray_pos, vecd3 ray_dir) const {
	collision c;
	
//	if (debug) printf("   Ray Marching (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f)\n", ray_pos.x, ray_pos.y, ray_pos.z, ray_dir.x, ray_dir.y, ray_dir.z);
	// For each iteration...
	for (int i = 0; i < 3200; i++) {
		// Get the distance to the scene.
		distance_to_object d = distance(ray_pos);
		
//		if (debug) printf("    %d: |(%.2f, %.2f, %.2f)| = %.2f\n", i, ray_pos.x, ray_pos.y, ray_pos.z, d.dis);
		
		// Detect collision.
		if (d.dis < COLLISION_DIST) {
			c.pos = ray_pos;
			c.object_ind = d.o_i;
//			if (debug) printf("    Collided.\n");
			return c;
		}
		
		// If the distance is too large, quit.
		if (d.dis > MAX_RAY_DIS) {
//			if (debug) printf("    Escaped.\n");
			return c;
		}
		
		// Record the closest approach.
		if (d.dis < c.approach_min) {
			c.approach_min = d.dis;
			c.approach_pos = ray_pos;
		}
		
//		if (debug) printf("Advancing %.2f * (%.2f, %.2f, %.2f)\n", d.dis, ray_dir.x, ray_dir.y, ray_dir.z);
		
		// Advance the ray "dis" units.
		ray_pos.x += ray_dir.x * d.dis;
		ray_pos.y += ray_dir.y * d.dis;
		ray_pos.z += ray_dir.z * d.dis;
	}
	
	printf("Killing Secondary Ray\n");
	
	return c;
}

#ifdef GPU_ENABLED
__device__
#endif
vecd3 scene_descriptor::normal(vecd3 pos, int object_ind) const {
	vecd3 a(
		distance_to_individual(pos + vecd3(NORMAL_DIST, 0, 0), object_ind),
		distance_to_individual(pos + vecd3(0, NORMAL_DIST, 0), object_ind),
		distance_to_individual(pos + vecd3(0, 0, NORMAL_DIST), object_ind)
	);
//	printf("Done x3\n");
	
	vecd3 b(
		distance_to_individual(pos - vecd3(NORMAL_DIST, 0, 0), object_ind),
		distance_to_individual(pos - vecd3(0, NORMAL_DIST, 0), object_ind),
		distance_to_individual(pos - vecd3(0, 0, NORMAL_DIST), object_ind)
	);
//	printf("Done x3\n");
	
//	if (debug) printf("(%lf, %lf, %lf) - (%lf, %lf, %lf)\n", a.x, a.y, a.z, b.x, b.y, b.z);
	
	return (a - b).normalize();
}

scene_descriptor::~scene_descriptor() {
	#ifdef GPU_ENABLED
		cudaError_t err;
		
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Error prior to scene_descriptor destructor call: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		bool has_err = false;
		
		err = cudaFree(objects);
		if (err != cudaSuccess) {
			printf("Freeing Objects: %s\n", cudaGetErrorName(err));
			has_err = true;
		}
		
		err = cudaFree(params);
		if (err != cudaSuccess) {
			printf("Freeing Params: %s\n", cudaGetErrorName(err));
			has_err = true;
		}
		
		err = cudaFree(lights);
		if (err != cudaSuccess) {
			printf("Freeing Lights: %s\n", cudaGetErrorName(err));
			has_err = true;
		}
		
		if (has_err) {
			printf("Possible double free of 0x%p. Make sure you aren't making any copies of a scene_descriptor. Issues of that nature may cause the computer to freeze up due to memory leaks on the GPU.\n", this);
			exit(1);
		}
	#else
		delete[] objects;
		delete[] params;
		delete[] lights;
	#endif
}

/* ---- camera ---- */

camera::camera(vecd3 pos, quaternion rot, float fov_slope, int width, int height, int nsl) : pos(pos), rot(rot), fov(fov_slope), width(width), height(height), nsl(nsl), img(NULL), accl(NULL) {
	// cudaError_t err;
	
	int ray_kernel_width = nsl * 2 + 1;
	
	// Check that the width and height are valid.
	#ifdef GPU_ENABLED
		if (width % (ray_kernel_width * 8) != 0) {
			printf("Failed to initialize camera: With target GPU and nsl of %d, width must be divisible by %d.\n", nsl, ray_kernel_width * 8);
			return;
		}
		if (height % (ray_kernel_width * 8) != 0) {
			printf("Failed to initialize camera: With target GPU and nsl of %d, height must be divisible by %d.\n", nsl, ray_kernel_width * 8);
			return;
		}
	#else
		if (width % ray_kernel_width != 0) {
			printf("Failed to initialize camera: With target CPU and nsl of %d, width must be divisible by %d.\n", nsl, ray_kernel_width);
			return;
		}
		if (height % ray_kernel_width != 0) {
			printf("Failed to initialize camera: With target CPU and nsl of %d, height must be divisible by %d.\n", nsl, ray_kernel_width);
			return;
		}
	#endif
	
	// Allocate space for the image.
	#ifdef GPU_ENABLED
		cudaError_t err;
		
		err = cudaMallocManaged(&img, width*height*3);
		if (err != cudaSuccess) {
			printf("Allocating Image: %s\n", cudaGetErrorName(err));
		}
	#else
		img = new uint8_t[width*height*3];
	#endif
	
	// Allocate space for the acceleration structure
	#ifdef GPU_ENABLED
		err = cudaMallocManaged(&accl, width * height / ray_kernel_width / ray_kernel_width * sizeof(float));
		if (err != cudaSuccess) {
			printf("Allocating Image: %s\n", cudaGetErrorName(err));
		}
		
		// err = cudaGetLastError();
		// printf("Allocating Acceleration Structure: %s\n", cudaGetErrorName(err));
	#else
		accl = new float[width * height / ray_kernel_width / ray_kernel_width];
	#endif
}

// TODO: Calling this function multiple times on the same camera object causes a crash. Probably a destructor call hidden somewhere.
// I tried passing the camera by reference but it didn't help. Passing the camera by reference is probably completely unnecessary, but it has no performance impact.
void camera::render(const scene_descriptor& sd) {
	int ray_kernel_width = nsl * 2 + 1;
	
	if (img == NULL || accl == NULL) {
		printf("Error: Cannot render, memory uninitiaalized.\n");
		exit(1);
	}
	
	#ifdef GPU_ENABLED
		cudaError_t err;
		
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Error occurred prior to rendering: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		// Kernel will use 64 registers, so blocks must have no more than 512 threads or register usage will exceed 32,768 per SM.
		dim3 block(8, 8);
		dim3 prm_grid(width / ray_kernel_width / 8, height / ray_kernel_width / 8);
		dim3 scd_grid(width / 8, height / 8);
		
		// Move the scene descriptor to the GPU.
		scene_descriptor* sd_gpu;
		err = cudaMallocManaged(&sd_gpu, sizeof(scene_descriptor));
		if (err != cudaSuccess) {
			printf("Allocated Scene Descriptor: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		cudaMemcpy(sd_gpu, &sd, sizeof(scene_descriptor), cudaMemcpyHostToDevice);
		
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Moved Descriptor to Device: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		// Move the camera to the GPU.
		camera* cam_gpu;
		err = cudaMallocManaged(&cam_gpu, sizeof(scene_descriptor));
		if (err != cudaSuccess) {
			printf("Allocated Cmera on Device: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		cudaMemcpy(cam_gpu, this, sizeof(scene_descriptor), cudaMemcpyHostToDevice);
		
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Moved Descriptor to Device: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		// Render the scene.
		cast_primary<<<prm_grid, block>>>(*sd_gpu, *cam_gpu, nsl, width / ray_kernel_width, height / ray_kernel_width, accl);
		
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Launched Primary: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		cudaDeviceSynchronize();
		
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Synchronized Primary: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		cast_secondary<<<scd_grid, block>>>(*sd_gpu, *cam_gpu, nsl, width / ray_kernel_width, height / ray_kernel_width, img, accl);
		
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Launched Secondary: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		cudaDeviceSynchronize();
		
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("Synchronized Secondary: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		err = cudaFree(sd_gpu);
		if (err != cudaSuccess) {
			printf("Freeing Descriptor: %s\n", cudaGetErrorName(err));
			exit(1);
		}
		
		err = cudaFree(cam_gpu);
		if (err != cudaSuccess) {
			printf("Freeing Camera: %s\n", cudaGetErrorName(err));
			exit(1);
		}
	#else
		cast_primary(sd, *this, nsl, width / ray_kernel_width, height / ray_kernel_width, accl);
		cast_secondary(sd, *this, nsl, width / ray_kernel_width, height / ray_kernel_width, img, accl);
	#endif
}

camera::~camera() {
	#ifdef GPU_ENABLED
		cudaError_t err;
	#endif
	
	if (img == NULL) {
		#ifdef GPU_ENABLED
			err = cudaFree(img);
			if (err != cudaSuccess) {
				printf("Freeing Image: %s\n", cudaGetErrorName(err));
				exit(1);
			}
		#else
			delete[] img;
		#endif
	}
	if (accl == NULL) {
		#ifdef GPU_ENABLED
			err = cudaFree(accl);
			if (err != cudaSuccess) {
				printf("Freeing Acceleration Structure: %s\n", cudaGetErrorName(err));
				exit(1);
			}
		#else
			delete[] accl;
		#endif
	}
}
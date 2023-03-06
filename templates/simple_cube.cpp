#include <stdio.h>
#include <stdint.h>

#include <time.h>

// Class definitions specific to this project.
#include "ray_march.hpp"

using namespace util;

int main() {
	int width = 480;
	int height = 480;
	
	int num_secondary_layers = 1;
	
	// Create the scene descriptor.
	scene_descriptor sd(2, 6, 4);
	
	// Set ambient lighting.
	sd.set_ambience(20, rgb(255, 255, 255));
	
	float params[] = {-1, 1, -1, 1, -1, 1};
	
	// Add a simple sphere to the scene descriptor.
	sd.add_object_with_params(&INFINITE_PLANE_CPU, &PHONG_CPU, vecd3(0, 0, 0), qid, rgb(255, 255, 255), rgb(255, 255, 255), 1, INFINITE_PLANE_N, NULL);
	sd.add_object_with_params(&CUBE_CPU, &PHONG_CPU, vecd3(0, 0, 1), qid, rgb(255, 255, 255), rgb(255, 255, 255), 20, CUBE_N, params);
	
	// Add lights to the scene.
	sd.add_light(vecd3(1, -2, 1), 1, 70, rgb(255, 255, 255), 70, rgb(255, 255, 255), true);
	sd.add_light(vecd3(1, 1, 1), 1, 70, rgb(255, 255, 255), 70, rgb(255, 255, 255), true);
	sd.add_light(vecd3(0, 0, 3.5), 0.45, 3000, rgb(80, 255, 255), 3000, rgb(100, 255, 255), false);
	
	// Create the camera.
	camera cam(vecd3(-6, 6, 8), quaternion(vecd3(0, 0, 1), -3.14159/4) * quaternion(vecd3(0, 1, 0), 3.14159/4), 1, width, height, num_secondary_layers);
	
	// Render the scene.
	cam.render(sd);
	
	// Save the image to file.
	char hdr[64];
	int hdr_len = sprintf(hdr, "P6 %d %d 255 ", width, height);
	
	FILE* fout = fopen("out.ppm", "wb");
	fwrite(hdr, 1, hdr_len, fout);
	fwrite(cam.img, 1, width*height*3, fout);
	fclose(fout);
	
	printf("Goodbye!\n");
	
	return 0;
}
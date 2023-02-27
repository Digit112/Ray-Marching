#include <stdio.h>
#include <stdint.h>

#include "utility/vec.hpp"

// Class definitions specific to this project.
#include "ray_march.hpp"

using namespace util;

int main() {
	int width = 1080/2;
	int height = 1080/2;
	
	// Create the scene descriptor.
	scene_descriptor sd(2, 6, 1);
	
	// Set ambient lighting.
	sd.set_ambience(30, rgb(255, 255, 255));
	
	float params[] = {-1, 1, -1, 1, -1, 1};
	
	// Add a simple sphere to the scene descriptor.
	sd.add_object_with_params(INFINITE_PLANE, vecd3(0, 0, 0), qid, rgb(255, 255, 255), rgb(255, 255, 255), 1, INFINITE_PLANE_N, NULL);
	sd.add_object_with_params(CUBE, vecd3(0, 0, 1), qid, rgb(255, 255, 255), rgb(255, 255, 255), 15, CUBE_N, params);
	
	// Add lights to the scene.
	sd.add_light(vecd3(2, -2, 3), 1200, rgb(255, 255, 255), 1200, rgb(255, 255, 255));
	
	// Create the camera.
	camera cam(vecd3(-6, 2.5, 7), quaternion(vecd3(0, 0, 1), -3.14159/8) * quaternion(vecd3(0, 1, 0), 3.14159/4), 1);
	
	// Create the image buffer
	uint8_t* img = new uint8_t[width*height*3];
	
	// Render the scene.
	cam.render(sd, width, height, img);
	
	// Save the image to file.
	char hdr[64];
	int hdr_len = sprintf(hdr, "P6 %d %d 255 ", width, height);
	
	FILE* fout = fopen("out.ppm", "wb");
	fwrite(hdr, 1, hdr_len, fout);
	fwrite(img, 1, width*height*3, fout);
	fclose(fout);
	
	delete[] img;
	
	printf("Goodbye!\n");
	
	return 0;
}
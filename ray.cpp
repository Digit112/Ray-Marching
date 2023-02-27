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
	scene_descriptor sd(2, 6, 3);
	
	// Set ambient lighting.
	sd.set_ambience(20, rgb(255, 200, 255));
	
	float params[] = {-1, 1, -1, 1, -1, 1};
	
	// Add a simple sphere to the scene descriptor.
	sd.add_object_with_params(INFINITE_PLANE, vecd3(0, 0, 0), qid, rgb(255, 255, 255), rgb(255, 255, 255), 1, INFINITE_PLANE_N, NULL);
	sd.add_object_with_params(CUBE, vecd3(0, 0, 1), qid, rgb(255, 255, 255), rgb(255, 255, 255), 8, CUBE_N, params);
	
	// Add lights to the scene.
	sd.add_light(vecd3(-0.75, -4, 3), 0.5, 2000, rgb(255, 0, 60), 1200, rgb(255, 60, 80));
	sd.add_light(vecd3(4, 0.75, 3), 0.5, 2000, rgb(60, 0, 255), 1200, rgb(80, 60, 255));
	sd.add_light(vecd3(-5, 5, 3), 0.5, 1000, rgb(255, 0, 255), 800, rgb(255, 60, 255));
	// Create the camera.
	camera cam(vecd3(-6, 6, 8), quaternion(vecd3(0, 0, 1), -3.14159/4) * quaternion(vecd3(0, 1, 0), 3.14159/4), 1);
	
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
#include <stdio.h>
#include <stdint.h>

#include "utility/vec.hpp"

// Class definitions specific to this project.
#include "ray_march.hpp"

using namespace util;

int main() {
	int width = 1080/2;
	int height = 1080/2;
	
	float radius = 1;
	
	// Create the scene descriptor. This can hold one object, one parameter, and three lights.
	scene_descriptor sd(1, 1, 3);
	
	// Set ambient lighting.
	sd.set_ambience(10, rgb(255, 255, 255));
	
	// Add a simple sphere to the scene descriptor.
	sd.add_object_with_params(SPHERE, vecd3(0, 0, 0), qid, rgb(255, 255, 255), rgb(255, 255, 255), 20, SPHERE_N, &radius);
	
	// Add lights to the scene.
	sd.add_light(vecd3(-0.5, -2, 5), 9600, rgb(255, 0, 0), 9600, rgb(255, 60, 60));
	sd.add_light(vecd3(-0.5, 2, 5), 9600, rgb(0, 0, 255), 9600, rgb(60, 60, 255));
	sd.add_light(vecd3(-4, 0, -2), 1000, rgb(255, 0, 255), 1000, rgb(255, 60, 255));
	
	// Create the camera. "qid" -> "Quaternion Identity" -> "No rotation"
	camera cam(vecd3(-4, 0, 0), qid, 1);
	
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
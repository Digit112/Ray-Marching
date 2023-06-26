#include <stdio.h>
#include <stdint.h>

#include <time.h>

// Class definitions specific to this project.
#define GPU_ENABLED
#include "ray_march.hpp"

using namespace util;

int main() {
	int width = 1080*2;
	int height = 1080*2;
	
	int frames = 24*8;
	
	int num_secondary_layers = 0;
	
	for (int f = 180; f < frames; f++) {
		printf("Rendering frame %d...\n", f);
		
		// Create the scene descriptor.
		scene_descriptor sd(1, 1, 2);
		
		// Set ambient lighting.
		sd.set_ambience(10, rgb(255, 255, 255));
		
		float power = f / 24;
		
		// Add a simple sphere to the scene descriptor.
		sd.add_object_with_params(&SIERPINSKI_GPU, &PHONG_GPU, vecd3(0, 0, (float) 1/3), qid, rgb(255, 255, 255), rgb(255, 255, 255), 1, SIERPINSKI_N, &power);
		
		float cam_dis = 2.5;
		
		// Create the camera.
		float theta = (float) f / frames * (2 * 3.1415962) + 3.1415962/2;
		quaternion rot(vecd3(0, 0, 1), theta);
		camera cam(rot.apply(vecd3(-cam_dis, 0, (float) 1/2)), quaternion(vecd3(0, 0, 1), theta), 1, width, height, num_secondary_layers);
		
		// Add lights to the scene.
		sd.add_light(-rot.apply(vecd3(0.5,  0,  1)), 1, 200, rgb(255, 255, 255), 200, rgb(255, 255, 255), true);
		sd.add_light(-rot.apply(vecd3(0.5,  0,  -1)), 1, 200, rgb(255, 255, 255), 200, rgb(255, 255, 255), true);
		
		clock_t start = clock();
		
		cam.render(sd);
		
		clock_t end = clock();
		
		printf("Rendered in %.4fs\n", ((float) end - start) / CLOCKS_PER_SEC);
		
		// Save the image to file.
		char hdr[64];
		int hdr_len = sprintf(hdr, "P6 %d %d 255 ", width, height);
		
		char fn[64];
		sprintf(fn, "frames/%03d.ppm", f);
		
		FILE* fout = fopen(fn, "wb");
		fwrite(hdr, 1, hdr_len, fout);
		fwrite(cam.img, 1, width*height*3, fout);
		fclose(fout);
		
		// Sleep for a brief time to reduce load on the system. The computer may halt the program otherwise.
		clock_t timer = clock();
		while ((float) (clock() - timer) / CLOCKS_PER_SEC < 0.25)  {}
	}
	
	printf("Goodbye!\n");
	
	return 0;
}
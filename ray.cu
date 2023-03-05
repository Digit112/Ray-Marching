#include <stdio.h>
#include <stdint.h>

#include <time.h>

// Class definitions specific to this project.
#define GPU_ENABLED
#include "ray_march.hpp"

using namespace util;

int main() {
	int width = 3120 * 2;
	int height = 3120 * 2;
	
	int frames = 768;
	
	int num_secondary_layers = 1;
	
	for (int f = 679; f < frames; f++) {
		printf("Rendering frame %d...\n", f);
		
		// Create the scene descriptor.
		scene_descriptor sd(1, 1, 4);
		
		// Set ambient lighting.
		sd.set_ambience(10, rgb(255, 255, 255));
		
		float power = -sin((float) f / frames * 2 * 3.141596) * 3 + 5;
		
		// Add a simple sphere to the scene descriptor.
		sd.add_object_with_params(&MANDELBULB_GPU, &MANDELBULB_SHADER_GPU, vecd3(0, 0, 0), qid, rgb(255, 255, 255), rgb(255, 255, 255), 1, MANDELBULB_N, &power);
		
		// Add lights to the scene.
		sd.add_light(vecd3( 1,  1,  1), 1, 200, rgb(255, 255, 255), 200, rgb(255, 255, 255), true);
		sd.add_light(vecd3( 1, -1,  1), 1, 200, rgb(255, 255, 255), 200, rgb(255, 255, 255), true);
		sd.add_light(vecd3(-1,  1,  1), 1, 200, rgb(255, 255, 255), 200, rgb(255, 255, 255), true);
		sd.add_light(vecd3(-1, -1,  1), 1, 200, rgb(255, 255, 255), 200, rgb(255, 255, 255), true);
		
		float cam_dis = 3 + sin((float) f / frames * 2 * 3.141596) * 0.6 + 0.6;
		
		// Create the camera.
		float theta = (float) f / frames * (4 * 3.1415962) + 3.1415962/2;
		camera cam(quaternion(vecd3(0, 0, 1), theta).apply(vecd3(-cam_dis, 0, 0)), quaternion(vecd3(0, 0, 1), theta), 1, width, height, num_secondary_layers);
		
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
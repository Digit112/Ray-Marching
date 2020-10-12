#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <CL/cl.h>

#include "ppmlib.h"

#define BUF1_S 1024
#define BUF4_S 16384

int main() {
	size_t width = 1024;
	size_t height = 1024;
	
	int errcd;
	FILE* fp;
	char* buf1;
	char* buf4;
	
	int start = 0;
	int end = 240;
	int frames = 240;
	
	cl_platform_id platform;
	errcd = clGetPlatformIDs(1, &platform, NULL);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	cl_device_id device;
	errcd = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	buf1 = malloc(BUF1_S);
	
	errcd = clGetDeviceInfo(device, CL_DEVICE_NAME, BUF1_S, buf1, NULL);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	printf("%s\n", buf1);
	
	free(buf1);
	
	cl_context context;
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &errcd);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	buf4 = malloc(BUF4_S);
	fp = fopen((char*) "rm.cl", "r");
	size_t source_size;
	source_size = fread(buf4, 1, BUF4_S, fp);
	buf4[source_size] = '\0';
	fclose(fp);
	printf("Read %ld bytes of code into buf4.\n", source_size);
	
	cl_program program;
	program = clCreateProgramWithSource(context, 1, (const char**) &buf4, &source_size, &errcd);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	free(buf4);
	
	errcd = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	if (errcd == CL_BUILD_PROGRAM_FAILURE || errcd == -9999) {
		size_t log_size;
		char* log;
		
		errcd = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
		log = malloc(log_size);
		errcd = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
		printf("%s\n", log);
		free(log);
	}	
	
	cl_kernel ray_march;
	ray_march = clCreateKernel(program, (char*) "ray_march", &errcd);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	cl_command_queue queue;

#ifdef CL_VERSION_2_0
	queue = clCreateCommandQueueWithProperties(context, device, NULL, &errcd);
#else
	queue = clCreateCommandQueue(context, device, 0, &errcd);
#endif

	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	/* SETUP COMPLETE */
	
	// Create image output
	cl_mem imbuf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width*height*4*sizeof(unsigned short), NULL, &errcd);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	cl_image_format imform = {CL_RGBA, CL_UNSIGNED_INT16};
	cl_image_desc imdesc = {CL_MEM_OBJECT_IMAGE2D, width, height, 0, 0, 0, 0, 0, 0, imbuf};
	
	cl_mem out = clCreateImage(context, CL_MEM_WRITE_ONLY, &imform, &imdesc, NULL, &errcd);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	// Set argument
	errcd = clSetKernelArg(ray_march, 0, sizeof(size_t), &width);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	errcd = clSetKernelArg(ray_march, 1, sizeof(size_t), &height);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	errcd = clSetKernelArg(ray_march, 2, sizeof(cl_mem), &out);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	size_t* gws = malloc(2 * sizeof(size_t));
	gws[0] = width;
	gws[1] = height;
	
	struct img im = {width, height, NULL, 6, 255};
	im.data = malloc(width * height * 4 * sizeof(unsigned short));
	
	size_t* origin = malloc(3*sizeof(size_t));
	size_t* region = malloc(3*sizeof(size_t));
	
	origin[0] = 0; origin[1] = 0; origin[2] = 0;
	region[0] = width; region[1] = height; region[2] = 1;
	
	char* filename_buf = malloc(32);
	
	double start_t = (double) clock() / CLOCKS_PER_SEC;
	
	double start_f;
	double rendr_f;
	double saved_f;
	
	float angle = 0;
	float power = 1.5;
	
	printf("Beginning calculations.\n");
	
	errcd = clSetKernelArg(ray_march, 3, sizeof(float), &angle);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	errcd = clSetKernelArg(ray_march, 4, sizeof(float), &power);
	if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
	
	for (int f = start; f < end; f++) {
		start_f = (double) clock() / CLOCKS_PER_SEC;
		
		angle = (float) f / frames * 2 * 3.14159;
		power = (float) f / frames * 3 + 1;
		
		errcd = clSetKernelArg(ray_march, 3, sizeof(float), &angle);
		if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
//		errcd = clSetKernelArg(ray_march, 4, sizeof(float), &power);
//		if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
		
		errcd = clEnqueueNDRangeKernel(queue, ray_march, 2, NULL, gws, NULL, 0, NULL, NULL);
		if (errcd != CL_SUCCESS) printf("Error on line %d in %s: %d\n", __LINE__, __FILE__, errcd);
		
		clEnqueueReadImage(queue, out, CL_TRUE, origin, region, 0, 0, im.data, 0, NULL, NULL);
		
		rendr_f = (double) clock() / CLOCKS_PER_SEC;
		
		RGBAtoRGB(&im);
		
		sprintf(filename_buf, "out/%03d.ppm", f);
		imgsave(im, filename_buf);
		
		saved_f = (double) clock() / CLOCKS_PER_SEC;
		
		printf("Frame %d r, s, t: %.2fs, %.2fs, %.2fs\n", f, rendr_f - start_f, saved_f - rendr_f, saved_f - start_f);
	}
	
	double end_t = (double) clock() / CLOCKS_PER_SEC;
	
	printf("Rendered in %.4f seconds.\n", end_t-start_t);
	
	free(im.data);
	
	free(gws);
	free(origin);
	free(region);
	
	return 0;
}

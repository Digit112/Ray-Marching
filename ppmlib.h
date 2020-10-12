#ifndef ppmlib
#define ppm_lib

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct img {
	size_t width;
	size_t height;
	
	unsigned short* data;
	
	unsigned char flags;
	
	unsigned short max_val;
};

// Flags:
//       ASCII Bin
// Bit     1    4
// Gray    2    5
// Color   3    6

// Any value not in [1, 6] will cause an error

void imgsave(const struct img out, const char* fn) {
	if (out.flags < 1 || out.flags > 6) {
		printf("Can't save image with invalid flags %hhx.\n", out.flags);
		return;
	}
	
	if (out.width < 1 || out.height < 1) {
		printf("Can't save image with invalid size %ldx%ld.\n", out.width, out.height);
		return;
	}
	
	// The width of the decimal representations of these values
	char* tempbuf = malloc(32);
	sprintf(tempbuf, "%hu", out.max_val);
	int max_val_width = strlen(tempbuf);
	sprintf(tempbuf, "%ld", out.width);
	int width_width = strlen(tempbuf);
	sprintf(tempbuf, "%ld", out.height);
	int height_width = strlen(tempbuf);
	
	size_t size = out.width * out.height;
	
	size_t filesize;
	
	int is_ascii = 1;
	
	unsigned char* outbuf = NULL;
	if (out.flags == 1) {
		filesize = width_width + height_width + 5 + 2*size + out.height;
	}
	else if (out.flags == 2) {
		filesize = width_width + height_width + max_val_width + 6 + (max_val_width + 1)*size + out.height;
	}
	else if (out.flags == 3) {
		filesize = width_width + height_width + max_val_width + 6 + (max_val_width + 1)*3*size + out.height;
	}
	else if (out.flags == 6) {
		is_ascii = 0;
		filesize = width_width + height_width + 10 + 3*size;
	}
	else {
		printf("P4 and P5 file formats not currently supported :(\n");
		return;
	}
	
	outbuf = malloc(filesize);
	if (outbuf == NULL) {
		printf("Failed to malloc memory required to save output image.\n");
		return;
	}
	
	size_t cursor;
	cursor = sprintf(outbuf, "P%hhu\n%ld %ld\n", out.flags, out.width, out.height);
	if (out.flags == 2 || out.flags == 3) {
		cursor += sprintf(outbuf+cursor, "%hu\n", out.max_val);
	}
	if (out.flags == 6) {
		cursor += sprintf(outbuf+cursor, "255\n");
	}
	
	if (is_ascii) {
		size_t i = 0;
		for (size_t y = 0; y < out.height; y++) {
			for (size_t x = 0; x < out.width; x++) {
				cursor += sprintf(outbuf+cursor, "%hu ", out.data[i]);
				i++;
				if (out.flags == 3) {
					cursor += sprintf(outbuf+cursor, "%hu ", out.data[i]);
					i++;
					cursor += sprintf(outbuf+cursor, "%hu ", out.data[i]);
					i++;
				}
			}
			sprintf(outbuf+cursor, (char*) "\n");
			cursor++;
		}
	}
	else {
		for (int i = 0; i < size*3; i++) {
			*(outbuf+cursor) = (unsigned char) out.data[i];
			cursor++;
		}
	}
	
	FILE* fp = fopen(fn, "wb+");
	fwrite(outbuf, 1, cursor, fp);
	fclose(fp);
	
	free(outbuf);
}

void RGBAtoRGB(struct img* out) {
	size_t size = out->width*out->height;
	for (size_t i = 0; i < size; i++) {
		out->data[i*3  ] = out->data[i*4  ];
		out->data[i*3+1] = out->data[i*4+1];
		out->data[i*3+2] = out->data[i*4+2];
	}
}

























#endif

# Ray-Marching
Ray Marching rendering engine. I use it to render fractals.

rm.cl is OpenCL C code which compiles at runtime for the specific GPU and Driver that you have.

rm.c is simply the code that runs the cl code. Compatible with OpenCL 1.2 and 2.x

All frames are rendered as ppm files in the ./out folder. The can be combined into video separately.

The program is controlled by parameters that exist both in the cl and c code.

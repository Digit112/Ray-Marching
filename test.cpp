#include <stdio.h>

#include "utility/util.hpp"

using namespace util;

int main() {
	vecd3 a(-1, 1, 1);
	vecd3 n(0, 0, 1);
	
	vecd3 b = vecd3::reflect(a, n);
	
	printf("%.2f, %.2f, %.2f\n", b.x, b.y, b.z);
	
	return 0;
}
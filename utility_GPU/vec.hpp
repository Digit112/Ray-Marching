#ifndef util_vec
#define util_vec

#include <math.h>
#include <stdio.h>

namespace util {
	class vecd2;
	class vecd3;
	class vecd4;
	class veci2;
	class veci3;
	class veci4;
	
	class quaternion;

	class vecd2 {
	public:
		double x;
		double y;
		
		__device__ __host__ vecd2();
		__device__ __host__ vecd2(double x, double y);
		
		// Returns the magnitude of this vector
		__device__ __host__ double mag() const;
		// Returns the squared magnitude of this vector
		__device__ __host__ double sqr_mag() const;
		
		// Returns the argument of the vector.
		__device__ __host__ double arg();
		
		__device__ __host__ vecd2 operator+(const vecd2& a) const;
		__device__ __host__ vecd2 operator-(const vecd2& a) const;
		__device__ __host__ vecd2 operator*(const vecd2& a) const;
		__device__ __host__ vecd2 operator/(const vecd2& a) const;
		__device__ __host__ vecd2 operator*(double a) const;
		__device__ __host__ vecd2 operator/(double a) const;
		__device__ __host__ vecd2 operator-() const;
		
		__device__ __host__ bool operator==(vecd2 a);
		
		// Returns whether this vector is nan. Only returns true if all elements are nan
		__device__ __host__ bool is_nan();
		
		// Returns a normalized version of this vector
		__device__ __host__ vecd2 normalize();
		__device__ __host__ vecd2 normalize(double t);
		
		// Returns the Dot Product of two vectors
		__device__ __host__ static double dot(vecd2 a, vecd2 b);	
	};

	class veci2 {
	public:
		int x;
		int y;
		
		__device__ __host__ veci2();
		__device__ __host__ veci2(int x, int y);
		
		// Returns the magnitude of this vector
		__device__ __host__ double mag() const;
		// Returns the square magnitude of this vector
		__device__ __host__ int sqr_mag() const;
		
		// Returns the argument of the vector.
		__device__ __host__ double arg();
		
		__device__ __host__ veci2 operator+(const veci2& a) const;
		__device__ __host__ veci2 operator-(const veci2& a) const;
		__device__ __host__ veci2 operator*(const veci2& a) const;
		__device__ __host__ veci2 operator/(const veci2& a) const;
		__device__ __host__ veci2 operator*(int a) const;
		__device__ __host__ veci2 operator/(int a) const;
		__device__ __host__ veci2 operator-() const;
		
		__device__ __host__ bool operator==(veci2 a);
		
		// Returns a normalized version of this vector
		__device__ __host__ vecd2 normalize();
		__device__ __host__ vecd2 normalize(double t);
		
		__device__ __host__ static int dot(vecd2 a, vecd2 b);	
	};
	
	class vecd3 {
	public:
		double x;
		double y;
		double z;
		
		__device__ __host__ vecd3();
		__device__ __host__ vecd3(double x, double y, double z);
		
		__device__ __host__ double mag() const;
		__device__ __host__ double sqr_mag() const;
		
		__device__ __host__ vecd3 operator+(const vecd3& a) const;
		__device__ __host__ vecd3 operator-(const vecd3& a) const;
		__device__ __host__ vecd3 operator*(const vecd3& a) const;
		__device__ __host__ vecd3 operator/(const vecd3& a) const;
		__device__ __host__ vecd3 operator*(double a) const;
		__device__ __host__ vecd3 operator/(double a) const;
		__device__ __host__ vecd3 operator-() const;
		
		__device__ __host__ bool operator==(vecd3 a);
		
		// Returns whether this vector is nan. Only returns true if all elements are nan.
		__device__ __host__ bool is_nan();
		
		__device__ __host__ vecd3 normalize();
		__device__ __host__ vecd3 normalize(double t);
		
		__device__ __host__ static double dot(vecd3 a, vecd3 b);
		__device__ __host__ static vecd3 cross(vecd3 a, vecd3 b);
		
		// Reflect the vector "a" across "b".
		// rev_reflect does the same, but also reverses the direction of the reflected vector.
		// Thus, rev_reflect returns the direction vector "a" is traveling after reflecting off of a surface with normal "b".
		__device__ __host__ static vecd3 reflect(vecd3 a, vecd3 b);
		__device__ __host__ static vecd3 rev_reflect(vecd3 a, vecd3 b);
	};
	
	class veci3 {
	public:
		int x;
		int y;
		int z;
		
		__device__ __host__ veci3();
		__device__ __host__ veci3(int x, int y, int z);
		
		__device__ __host__ double mag() const;
		__device__ __host__ int sqr_mag() const;
		
		__device__ __host__ veci3 operator+(const veci3& a) const;
		__device__ __host__ veci3 operator-(const veci3& a) const;
		__device__ __host__ veci3 operator*(const veci3& a) const;
		__device__ __host__ veci3 operator/(const veci3& a) const;
		__device__ __host__ veci3 operator*(int a) const;
		__device__ __host__ veci3 operator/(int a) const;
		__device__ __host__ veci3 operator-() const;
		
		__device__ __host__ bool operator==(veci3 a);
		
		__device__ __host__ vecd3 normalize();
		__device__ __host__ vecd3 normalize(double t);
		
		__device__ __host__ static int dot(veci3 a, veci3 b);
		__device__ __host__ static veci3 cross(veci3 a, veci3 b);
	};
	
	class vecd4 {
	public:
		double w;
		double x;
		double y;
		double z;
		
		__device__ __host__ vecd4();
		__device__ __host__ vecd4(double w, double x, double y, double z);
		
		__device__ __host__ double mag() const;
		__device__ __host__ double sqr_mag() const;
		
		__device__ __host__ vecd4 operator+(const vecd4& a) const;
		__device__ __host__ vecd4 operator-(const vecd4& a) const;
		__device__ __host__ vecd4 operator*(const vecd4& a) const;
		__device__ __host__ vecd4 operator/(const vecd4& a) const;
		__device__ __host__ vecd4 operator*(double a) const;
		__device__ __host__ vecd4 operator/(double a) const;
		__device__ __host__ vecd4 operator-() const;
		
		__device__ __host__ bool operator==(vecd4 a);
		
		// Returns whether this vector is nan. Only returns true if all elements are nan
		__device__ __host__ bool is_nan();
		
		__device__ __host__ vecd4 normalize();
		__device__ __host__ vecd4 normalize(double t);
		
		__device__ __host__ static double dot(vecd4 a, vecd4 b);
	};
	
	class veci4 {
	public:
		int w;
		int x;
		int y;
		int z;
		
		__device__ __host__ veci4();
		__device__ __host__ veci4(int w, int x, int y, int z);
		
		__device__ __host__ double mag() const;
		__device__ __host__ int sqr_mag() const;
		
		__device__ __host__ veci4 operator+(const veci4& a) const;
		__device__ __host__ veci4 operator-(const veci4& a) const;
		__device__ __host__ veci4 operator*(const veci4& a) const;
		__device__ __host__ veci4 operator/(const veci4& a) const;
		__device__ __host__ veci4 operator*(int a) const;
		__device__ __host__ veci4 operator/(int a) const;
		__device__ __host__ veci4 operator-() const;
		
		__device__ __host__ bool operator==(veci4 a);
		
		__device__ __host__ vecd4 normalize();
		__device__ __host__ vecd4 normalize(double t);
		
		__device__ __host__ static int dot(veci4 a, veci4 b);
	};
	
	class complex : public vecd2 {
	public:
		__device__ __host__ complex();
		__device__ __host__ complex(double r, double i);
		__device__ __host__ complex(double theta);
		
		__device__ __host__ complex(const vecd2&);
		
		__device__ __host__ complex operator*(const complex& a) const;
		__device__ __host__ complex operator*(double a) const;
		
		__device__ __host__ static complex scale(const complex& a, const complex& b);
		
		// Complex conjugation
		__device__ __host__ complex operator~() const;
	};
	
	class quaternion : public vecd4 {
	public:
		__device__ __host__ quaternion();
		__device__ __host__ quaternion(double w, double x, double y, double z);
		__device__ __host__ quaternion(vecd3 axis, double theta);
		
		__device__ __host__ quaternion(const vecd4& a);
		
		__device__ __host__ quaternion operator=(vecd4) const;
		
		__device__ __host__ quaternion operator*(const quaternion& a) const; // Hamilton product
		__device__ __host__ quaternion operator*(double a) const;
		
		__device__ __host__ static quaternion scale(const quaternion& a, const quaternion& b);
		
		__device__ __host__ bool operator==(veci4 a);
		
		// Complex conjugation
		__device__ __host__ quaternion operator~() const;
		
		__device__ __host__ static quaternion hamilton(const quaternion& a, const quaternion& b);
		__device__ __host__ static vecd3 vhamilton(const quaternion& a, const quaternion& b);
		
		__device__ __host__ quaternion& mhamilton(quaternion& a, const quaternion& b);
		
		// Normalize identical to the vecd4 normalize. This version exists so that quaternion::normalize() will return a quaternion.
		__device__ __host__ quaternion normalize();
		__device__ __host__ quaternion normalize(double t);
		
		// Rotates the passed vector according to the current quaternion about the origin.
		__device__ __host__ vecd3 apply(const vecd3& in) const;
		
		// Rotate the given vector theta radians around the given axis or by applying the given quaternion.
		// Use offset to specify a point through which the axis of rotation passes.
		__device__ __host__ static vecd3 rotate(vecd3 in, vecd3 axis_offset, vecd3 axis_dir, double theta);
		__device__ __host__ static vecd3 rotate(vecd3 in, vecd3 axis_offset, quaternion r);
		
		// Interpolate between two quaternions in such a way as to produce smooth rotations.
		__device__ __host__ static quaternion slerp(const quaternion& a, const quaternion& b, double t); 
	};

	const vecd3 forward(1, 0, 0);
	const vecd3 backward(-1, 0, 0);
	const vecd3 right(0, 1, 0);
	const vecd3 left(0, -1, 0);
	const vecd3 up(0, 0, 1);
	const vecd3 down(0, 0, -1);
	
	const complex cid(1, 0);
	const quaternion qid(1, 0, 0, 0);
}


#include "vec.cpp"

#endif

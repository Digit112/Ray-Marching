double sphere(double3 p, double3 o, double r) {
	return distance(p, o) - r;
}

double cube(double3 p, double e2) {
	double3 q = {fabs(p.x) - e2, fabs(p.y) - e2, fabs(p.z) - e2};
	double3 m = {max(q.x, 0.0), max(q.y, 0.0), max(q.z, 0.0)};
	return length(m) + min(max(q.x, max(q.y, q.z)), 0.0);
}

double plane(double3 p, double3 n, double h) {
	return dot(p, normalize(n)) + h;
}

double xyplane(double3 p, double h) {
	return fabs(p.z - h);
}

double sierpinski(double3 p) {
	double r;
	
	double3 a = {1, 1, 1};
	
	const double scale = 2;
	
	int n;
	for (n = 0; n < 20; n++) {
		if (p.x + p.y < 0) p.xy = -p.yx;
		if (p.x + p.z < 0) p.xz = -p.zx;
		if (p.y + p.z < 0) p.zy = -p.yz;
		p = p*scale - a*(scale-1);
	}
	
	return length(p) * pow(scale, (double) -n);
}

double checker(double3 p) {
	double3 center = {0, 0, 0};
	double3 XYZ = { 1.0/3,  1.0/3,  1.0/3};
	double3 XYz = { 1.0/3,  1.0/3, -1.0/3};
	double3 XyZ = { 1.0/3, -1.0/3,  1.0/3};
	double3 Xyz = { 1.0/3, -1.0/3, -1.0/3};
	double3 xYZ = {-1.0/3,  1.0/3,  1.0/3};
	double3 xYz = {-1.0/3,  1.0/3, -1.0/3};
	double3 xyZ = {-1.0/3, -1.0/3,  1.0/3};
	double3 xyz = {-1.0/3, -1.0/3, -1.0/3};
	
	double3 c;
	
	int n;
	
	double dist;
	bool is_center;
	double d;
	for (n = 0; n < 5; n++) {
		is_center = true; c = center;
		dist = cube(p, 1.0/3);
		d = cube(p - XYZ, 1.0/3); if (d < dist) {dist = d; is_center = false; c = XYZ * 3;}
		d = cube(p - XYz, 1.0/3); if (d < dist) {dist = d; is_center = false; c = XYz * 3;}
		d = cube(p - XyZ, 1.0/3); if (d < dist) {dist = d; is_center = false; c = XyZ * 3;}
		d = cube(p - Xyz, 1.0/3); if (d < dist) {dist = d; is_center = false; c = Xyz * 3;}
		d = cube(p - xYZ, 1.0/3); if (d < dist) {dist = d; is_center = false; c = xYZ * 3;}
		d = cube(p - xYz, 1.0/3); if (d < dist) {dist = d; is_center = false; c = xYz * 3;}
		d = cube(p - xyZ, 1.0/3); if (d < dist) {dist = d; is_center = false; c = xyZ * 3;}
		d = cube(p - xyz, 1.0/3); if (d < dist) {dist = d; is_center = false; c = xyz * 3;}
		p = 3*p - 2*c;
	}
	
	return dist * pow(3, (double) -(n+1));
}
		

double mandelbulb(const double3* p, float power) {
	double3 z = *p;
	double dr = 1;
	float r = 0;
	
	float theta;
	float phi;
	double zr;
	
	double3 zt;
	
	for (int i = 0; i < 20; i++) {
		r = length(z);
		if (r > 20) break;
		
		// Convert to polar coordinates
		theta = acos(z.z / r);
		phi = atan2(z.y, z.x);
		dr = pow(r, power-1) * power * dr + 1;
		
		// Scale and rotate the point
		zr = pow(r, power);
		theta = theta*power;
		phi = phi*power;
		
		// convert back to cartesian coordinates
		double sin_theta = sin(theta);
		zt.x = sin_theta*cos(phi);
		zt.y = sin(phi)*sin_theta;
		zt.z = cos(theta);
		z = zr*zt;
		z += *p;
	}
	
	return 0.5*log(r)*r/dr;
}

double DE(double3 p) {
//	double3 o = {0, 0, -1};
//	return min(sphere(p, o, 0.5), xyplane(p, 1));
//	double3 n = {2.31, -2.31, -2.31};
	
	return checker(p);
}

double4 dhamiltonf(float4 a, float4 b) {
	double4 out = {a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y, a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x, a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w, a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z};
	return out;
}

double3 vdhamiltonf(float4 a, float4 b) {
	double3 out = {a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y, a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x, a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w};
	return out;
}

float4  fhamiltonf(float4 a, float4 b) {
	float4 out = {a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y, a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x, a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w, a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z};
	return out;
}

float3  vfhamiltonf(float4 a, float4 b) {
	float3 out = {a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y, a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x, a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w};
	return out;
}

double4 dhamiltond(double4 a, double4 b) {
	double4 out = {a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y, a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x, a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w, a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z};
	return out;
}

double3 vdhamiltond(double4 a, double4 b) {
	double3 out = {a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y, a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x, a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w};
	return out;
}

float4  fhamiltond(double4 a, double4 b) {
	float4 out = {a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y, a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x, a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w, a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z};
	return out;
}

float3  vfhamiltond(double4 a, double4 b) {
	float3 out = {a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y, a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x, a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w};
	return out;
}

kernel void ray_march(long width, long height, write_only image2d_t out, float angle, float power) {
	// Number of iterations to perform
	const int iter = 800;
	
	// Glowing edges
	const float glow = 0;
	const float max_glow = 120;
	
	// Threshhold to use for collision
	const double threshhold = 0.00001;
	const double super_threshhold = 0.0000001;
	
	// Sun
	float3 sun_dir = {-0.8, 0.4, -0.2};
	float sol_illum = 200;
	float amb_illum = 60;
	
	float penumbra_scale = 0.1;
	
	// Distance until considered hitting the sky
	float sky_dis = 20;
	
	// FOV of the camera
	const float theta_x = radians((float) 45);
	const float theta_y = theta_x * height / width;
	
	// Initial Camera position
	double3 pos = {-4, 0, 0};
	
	sun_dir = normalize(sun_dir);
	
	// Rotation to apply to camera (camera will be orbited about the origin)
	float3 cam_rot_axis = {1, 1, 0};
	float cam_rot_theta = radians((float) 0);
	
	float s;
	
	// Orbit the camera
	float3 cam_spin_axis = {0, 0, 1};
	cam_spin_axis = normalize(cam_spin_axis);
	s = sin(angle/2);
	float4 cam_spin_q = {cam_spin_axis.x * s, cam_spin_axis.y * s, cam_spin_axis.z * s, cos(angle/2)};
	
	// Scene origin
	const float3 origin = {0, 0, 0};
	
	cam_rot_axis = normalize(cam_rot_axis);
	
	s = sin(cam_rot_theta/2);
	float4 cam_rot_q = {cam_rot_axis.x * s, cam_rot_axis.y * s, cam_rot_axis.z * s, cos(cam_rot_theta/2)};
	
	cam_rot_q = fhamiltonf(cam_rot_q, cam_spin_q);
	
	float4 cam_rot_q_i = {-cam_rot_q.x, -cam_rot_q.y, -cam_rot_q.z, cam_rot_q.w};
	
	/* Begin Ray Marching */
	
	// Pixel for this work-item
	const int2 pix = {get_global_id(0), get_global_id(1)};
	
	// Determine the direction of this ray
	float3 dir = {1, tan((float) (pix.x - width/2) / (width/2) * (theta_x/2)), tan((float) (pix.y - height/2) / (height/2) * (theta_y/2))};
	dir = normalize(dir);
	float4 dir_q = {dir.x, dir.y, dir.z, 0};
	dir = vfhamiltonf(fhamiltonf(cam_rot_q, dir_q), cam_rot_q_i);
	
	// Determine initial position of this ray
	float4 pos_q = {pos.x, pos.y, pos.z, 0};
	pos = vdhamiltonf(fhamiltonf(cam_rot_q, pos_q), cam_rot_q_i);
	
	// Determine the position of the sun
	float4 sun_q = {sun_dir.x, sun_dir.y, sun_dir.z, 0};
	sun_dir = vfhamiltonf(fhamiltonf(cam_rot_q, sun_q), cam_rot_q_i);
	
	double3 cam_pos = pos;
	
	double3 prev_pos;
	double prev_dis;
	double dis = 0;
	
	double3 min_dis_p;
	double min_dis_0 = 1000;
	double min_dis_1 = 1000;
	
	bool did_hit_scene = false;
	bool did_hit_sky = true;
	
	double3 hit;
	double hit_dis;
		
	float shadow_mod = 1;
	
	int val;
	uint4 color; color.w = 255;
	
	int i;
	for (i = 0; i < iter; i++) {
		prev_dis = dis;
		dis = DE(pos);
		if (dis < min_dis_0) {
			min_dis_0 = dis;
			min_dis_p = pos;
		}
		
		if (dis < threshhold) {
			did_hit_scene = true;
			break;
		}
		if (distance(pos, cam_pos) > sky_dis) {
			break;
		}
		
		// Advance the ray
		prev_pos = pos;
		pos.x += dir.x * dis;
		pos.y += dir.y * dis;
		pos.z += dir.z * dis;
	}
	
	if (did_hit_scene) {
		// Attempt to linearly interpolate the location in which the ray crossed the threshold and move the position there.
		double t = (threshhold - dis) / (prev_dis - dis);
		pos = pos + t*(prev_pos - pos);
		hit = pos;
		hit_dis = DE(hit);
		
		int j;
		for (j = 0; j < iter; j++) {
			dis = DE(pos);
			min_dis_1 = min(min_dis_1, dis);
			
			if (dis < super_threshhold) {
				did_hit_sky = false;
				break;
			}
			if (distance(pos, cam_pos) > sky_dis) {
				did_hit_sky = true;
				break;
			}
			
			float d = penumbra_scale * (distance(pos, hit) + hit_dis);
			shadow_mod = min(shadow_mod, (float) (dis*(d+1) / (d*(dis+1))));
			
			// Advance the ray
			pos.x += sun_dir.x * dis;
			pos.y += sun_dir.y * dis;
			pos.z += sun_dir.z * dis;
		}
		
		if (dis < threshhold) {
			did_hit_sky = false;
		}
	}
	
	int R; int G; int B;
	if (did_hit_scene) {
		if (did_hit_sky) {
			val = shadow_mod * (sol_illum - amb_illum) + amb_illum;
		} else {
			val = amb_illum;
		}
		R = max(val * (cos(length(hit) * 5)          * 0.5 + 0.5), 8.0);
		G = 8;
		B = max(val * (cos(length(hit) * 5 + M_PI_F) * 0.5 + 0.5), 8.0);
	} else {
		val = (int) max(glow / (min_dis_0 + glow/max_glow), 8.0);
		R = val * (cos(length(min_dis_p) * 5)          * 0.5 + 0.5);
		G = 8;
		B = val * (cos(length(min_dis_p) * 5 + M_PI_F) * 0.5 + 0.5);
	}
	color.x = R; color.y = G; color.z = B;
	
	write_imageui(out, pix, color);
}

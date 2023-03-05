namespace util {
	/* ---- vecd2 ---- */
	
	vecd2::vecd2() : x(0), y(0) {}
	vecd2::vecd2(double x, double y) : x(x), y(y) {}
	
	double vecd2::mag() const {
		return sqrt(x*x + y*y);
	}
	
	double vecd2::sqr_mag() const {
		return x*x + y*y;
	}
	
	double vecd2::arg() {
		return atan2(y, x);
	}
	
	vecd2 vecd2::operator+(const vecd2& a) const {
		return vecd2(x+a.x, y+a.y);
	}
	vecd2 vecd2::operator-(const vecd2& a) const {
		return vecd2(x-a.x, y-a.y);
	}
	vecd2 vecd2::operator*(const vecd2& a) const {
		return vecd2(x*a.x, y*a.y);
	}
	vecd2 vecd2::operator/(const vecd2& a) const {
		return vecd2(x/a.x, y/a.y);
	}
	vecd2 vecd2::operator*(double a) const {
		return vecd2(x*a, y*a);
	}
	vecd2 vecd2::operator/(double a) const {
		return vecd2(x/a, y/a);
	}
	vecd2 vecd2::operator-() const {
		return vecd2(-x, -y);
	}
	
	bool vecd2::operator==(vecd2 a) {
		return x == a.x && y == a.y;
	}
	
	bool vecd2::is_nan() {
		return isnan(x) && isnan(y);
	}
	
	vecd2 vecd2::normalize() {
		double m = mag();
		return vecd2(x/m, y/m);
	}
	vecd2 vecd2::normalize(double t) {
		double m = mag() / t;
		return vecd2(x/m, y/m);
	}
	
	double vecd2::dot(vecd2 a, vecd2 b) {
		return a.x*b.x + a.y*b.y;
	}
	
	/* ---- veci2 ---- */
	
	veci2::veci2() : x(0), y(0) {}
	veci2::veci2(int x, int y) : x(x), y(y) {}
	
	double veci2::mag() const {
		return sqrt((float) x*x + y*y);
	}
	
	int veci2::sqr_mag() const {
		return x*x + y*y;
	}
	
	double veci2::arg() {
		return atan2((float) y, (float) x);
	}
	
	veci2 veci2::operator+(const veci2& a) const {
		return veci2(x+a.x, y+a.y);
	}
	veci2 veci2::operator-(const veci2& a) const {
		return veci2(x-a.x, y-a.y);
	}
	veci2 veci2::operator*(const veci2& a) const {
		return veci2(x*a.x, y*a.y);
	}
	veci2 veci2::operator/(const veci2& a) const {
		return veci2(x/a.x, y/a.y);
	}
	veci2 veci2::operator*(int a) const {
		return veci2(x*a, y*a);
	}
	veci2 veci2::operator/(int a) const {
		return veci2(x/a, y/a);
	}
	veci2 veci2::operator-() const {
		return veci2(-x, -y);
	}
	
	bool veci2::operator==(veci2 a) {
		return x == a.x && y == a.y;
	}
	
	vecd2 veci2::normalize() {
		double m = mag();
		return vecd2(x/m, y/m);
	}
	vecd2 veci2::normalize(double t) {
		double m = mag() / t;
		return vecd2(x/m, y/m);
	}
	
	int veci2::dot(vecd2 a, vecd2 b) {
		return a.x*b.x + a.y*b.y;
	}
	
	/* ---- vecd3 ---- */
	
	vecd3::vecd3() : x(0), y(0), z(0) {}
	vecd3::vecd3(double x, double y, double z) : x(x), y(y), z(z) {}
	
	double vecd3::mag() const {
		return sqrt(x*x + y*y + z*z);
	}
	
	double vecd3::sqr_mag() const {
		return x*x + y*y + z*z;
	}
	
	vecd3 vecd3::operator+(const vecd3& a) const {
		return vecd3(x+a.x, y+a.y, z+a.z);
	}
	vecd3 vecd3::operator-(const vecd3& a) const {
		return vecd3(x-a.x, y-a.y, z-a.z);
	}
	vecd3 vecd3::operator*(const vecd3& a) const {
		return vecd3(x*a.x, y*a.y, z*a.z);
	}
	vecd3 vecd3::operator/(const vecd3& a) const {
		return vecd3(x/a.x, y/a.y, z/a.z);
	}
	vecd3 vecd3::operator*(double a) const {
		return vecd3(x*a, y*a, z*a);
	}
	vecd3 vecd3::operator/(double a) const {
		return vecd3(x/a, y/a, z/a);
	}
	vecd3 vecd3::operator-() const {
		return vecd3(-x, -y, -z);
	}
	
	bool vecd3::operator==(vecd3 a) {
		return x == a.x && y == a.y && z == a.z;
	}
	
	bool vecd3::is_nan() {
		return isnan(x) && isnan(y) && isnan(z);
	}
	
	vecd3 vecd3::normalize() {
		double m = mag();
		return vecd3(x/m, y/m, z/m);
	}
	vecd3 vecd3::normalize(double t) {
		double m = mag() / t;
		return vecd3(x/m, y/m, z/m);
	}
	
	double vecd3::dot(vecd3 a, vecd3 b) {
		return a.x*b.x + a.y*b.y + a.z*b.z;
	}
	
	vecd3 vecd3::cross(vecd3 a, vecd3 b) {
		return vecd3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
	}
	
	vecd3 vecd3::reflect(vecd3 a, vecd3 b) {
		return b * vecd3::dot(a, b) * 2 - a;
	}
	
	vecd3 vecd3::rev_reflect(vecd3 a, vecd3 b) {
		return a - b * vecd3::dot(a, b) * 2;
	}
	
	/* ---- veci3 ---- */
	
	veci3::veci3() : x(0), y(0), z(0) {}
	veci3::veci3(int x, int y, int z) : x(x), y(y), z(z) {}
	
	double veci3::mag() const {
		return sqrt((float) x*x + y*y + z*z);
	}
	
	int veci3::sqr_mag() const {
		return x*x + y*y + z*z;
	}
	
	veci3 veci3::operator+(const veci3& a) const {
		return veci3(x+a.x, y+a.y, z+a.z);
	}
	veci3 veci3::operator-(const veci3& a) const {
		return veci3(x-a.x, y-a.y, z-a.z);
	}
	veci3 veci3::operator*(const veci3& a) const {
		return veci3(x*a.x, y*a.y, z*a.z);
	}
	veci3 veci3::operator/(const veci3& a) const {
		return veci3(x/a.x, y/a.y, z/a.z);
	}
	veci3 veci3::operator*(int a) const {
		return veci3(x*a, y*a, z*a);
	}
	veci3 veci3::operator/(int a) const {
		return veci3(x/a, y/a, z/a);
	}
	veci3 veci3::operator-() const {
		return veci3(-x, -y, -z);
	}
	
	bool veci3::operator==(veci3 a) {
		return x == a.x && y == a.y && z == a.z;
	}
	
	vecd3 veci3::normalize() {
		double m = mag();
		return vecd3(x/m, y/m, z/m);
	}
	vecd3 veci3::normalize(double t) {
		double m = mag() / t;
		return vecd3(x/m, y/m, z/m);
	}
	
	int veci3::dot(veci3 a, veci3 b) {
		return a.x*b.x + a.y*b.y + a.z*b.z;
	}
	
	veci3 veci3::cross(veci3 a, veci3 b) {
		return veci3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
	}
	
	/* ---- vecd4 ---- */
	
	vecd4::vecd4() : w(0), x(0), y(0), z(0) {}
	vecd4::vecd4(double w, double x, double y, double z) : w(w), x(x), y(y), z(z) {}
	
	double vecd4::mag() const {
		return sqrt(w*w + x*x + y*y + z*z);
	}
	
	double vecd4::sqr_mag() const {
		return w*w + x*x + y*y + z*z;
	}
	
	vecd4 vecd4::operator+(const vecd4& a) const {
		return vecd4(w+a.w, x+a.x, y+a.y, z+a.z);
	}
	vecd4 vecd4::operator-(const vecd4& a) const {
		return vecd4(w-a.w, x-a.x, y-a.y, z-a.z);
	}
	vecd4 vecd4::operator*(const vecd4& a) const {
		return vecd4(w*a.w, x*a.x, y*a.y, z*a.z);
	}
	vecd4 vecd4::operator/(const vecd4& a) const {
		return vecd4(w/a.w, x/a.x, y/a.y, z/a.z);
	}
	vecd4 vecd4::operator*(double a) const {
		return vecd4(w*a, x*a, y*a, z*a);
	}
	vecd4 vecd4::operator/(double a) const {
		return vecd4(w/a, x/a, y/a, z/a);
	}
	vecd4 vecd4::operator-() const {
		return vecd4(-w, -x, -y, -z);
	}
	
	bool vecd4::operator==(vecd4 a) {
		return w == a.w && x == a.x && y == a.y && z == a.z;
	}
	
	bool vecd4::is_nan() {
		return isnan(w) && isnan(x) && isnan(y) && isnan(z);
	}
	
	vecd4 vecd4::normalize() {
		double m = mag();
		return vecd4(w/m, x/m, y/m, z/m);
	}
	vecd4 vecd4::normalize(double t) {
		double m = mag() / t;
		return vecd4(w/m, x/m, y/m, z/m);
	}
	
	double vecd4::dot(vecd4 a, vecd4 b) {
		return a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z;
	}
	
	/* ---- veci4 ---- */
	veci4::veci4() : w(0), x(0), y(0), z(0) {}
	veci4::veci4(int w, int x, int y, int z) : w(w), x(x), y(y), z(z) {}
	
	double veci4::mag() const {
		return sqrt((float) w*w + x*x + y*y + z*z);
	}
	
	int veci4::sqr_mag() const {
		return w*w + x*x + y*y + z*z;
	}
	
	veci4 veci4::operator+(const veci4& a) const {
		return veci4(w+a.w, x+a.x, y+a.y, z+a.z);
	}
	veci4 veci4::operator-(const veci4& a) const {
		return veci4(w-a.w, x-a.x, y-a.y, z-a.z);
	}
	veci4 veci4::operator*(const veci4& a) const {
		return veci4(w*a.w, x*a.x, y*a.y, z*a.z);
	}
	veci4 veci4::operator/(const veci4& a) const {
		return veci4(w/a.w, x/a.x, y/a.y, z/a.z);
	}
	veci4 veci4::operator*(int a) const {
		return veci4(w*a, x*a, y*a, z*a);
	}
	veci4 veci4::operator/(int a) const {
		return veci4(w/a, x/a, y/a, z/a);
	}
	veci4 veci4::operator-() const {
		return veci4(-w, -x, -y, -z);
	}
	
	bool veci4::operator==(veci4 a) {
		return w == a.w && x == a.x && y == a.y && z == a.z;
	}
	
	vecd4 veci4::normalize() {
		double m = mag();
		return vecd4(w / m, x / m, y / m, z / m);
	}
	vecd4 veci4::normalize(double t) {
		double m = mag() / t;
		return vecd4(w/m, x/m, y/m, z/m);
	}
	
	int veci4::dot(veci4 a, veci4 b) {
		return a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z;
	}
	
	/* ---- complex ---- */
	
	complex::complex() : vecd2() {}
	complex::complex(double r, double i) : vecd2(r, i) {}
	complex::complex(double theta) : vecd2(cos(theta), sin(theta)) {}
	
	complex::complex(const vecd2& a) : vecd2(a) {}
	
	complex complex::operator*(const complex& a) const {
		return complex(x*a.x - y*a.y, x * a.y + y * a.x);
	}
	
	complex complex::operator*(double a) const {
		return complex(x*a, y*a);
	}
	
	complex complex::scale(const complex& a, const complex& b) {
		return complex(a.x*b.x, a.y*b.y);
	}
	
	complex complex::operator~() const {
		return complex(x, -y);
	}
	
	/* ---- quaternion ---- */
	
	quaternion::quaternion() : vecd4() {}
	quaternion::quaternion(double w, double x, double y, double z) : vecd4(w, x, y, z) {}

	quaternion::quaternion(vecd3 axis, double theta) {
		axis = axis.normalize();
		double s = sin(theta/2);
		w = cos(theta/2);
		x = axis.x * s;
		y = axis.y * s;
		z = axis.z * s;
	}
	
	quaternion::quaternion(const vecd4& a) : vecd4(a) {}
	
	quaternion quaternion::operator*(const quaternion& a) const {
		return quaternion(w*a.w - x*a.x - y*a.y - z*a.z, w*a.x + x*a.w + y*a.z - z*a.y, w*a.y - x*a.z + y*a.w + z*a.x, w*a.z + x*a.y - y*a.x + z*a.w);
	}
	
	quaternion quaternion::operator*(double a) const {
		return quaternion(w*a, x*a, y*a, z*a);
	}
	
	quaternion quaternion::scale(const quaternion& a, const quaternion& b) {
		return quaternion(a.w*b.w, a.x*b.x, a.y*b.y, a.z*b.z);
	}
	
	quaternion quaternion::operator=(vecd4 a) const {
		return quaternion(a.w, a.x, a.y, a.z);
	}
	
	quaternion quaternion::operator~() const {
		return quaternion(w, -x, -y, -z);
	}
	
	quaternion quaternion::normalize() {
		double m = mag();
		return quaternion(w/m, x/m, y/m, z/m);
	}
	
	quaternion quaternion::normalize(double t) {
		double m = mag() / t;
		return quaternion(w/m, x/m, y/m, z/m);
	}
	
	quaternion quaternion::hamilton(const quaternion& a, const quaternion& b) {
		return quaternion(a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z, a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y, a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x, a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w);
	}
	
	vecd3 quaternion::vhamilton(const quaternion& a, const quaternion& b) {
		return vecd3(a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y, a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x, a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w);
	}
	
	quaternion& quaternion::mhamilton(quaternion& a, const quaternion& b) {
		a.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
		a.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y;
		a.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x;
		a.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w;
		return a;
	}
	
	vecd3 quaternion::apply(const vecd3& in) const {
		return vhamilton(hamilton(*this, quaternion(0, in.x, in.y, in.z)), quaternion(this->w, -this->x, -this->y, -this->z));
	}
	
	vecd3 quaternion::rotate(vecd3 in, vecd3 axis_offset, vecd3 axis_dir, double theta) {
		in = in - axis_offset;
		in = quaternion(axis_dir, theta).apply(in);
		return in + axis_offset;
	}
	
	vecd3 quaternion::rotate(vecd3 in, vecd3 axis_offset, quaternion r) {
		in = in - axis_offset;
		in = r.apply(in);
		return in + axis_offset;
	}
	
	quaternion quaternion::slerp(const quaternion& a, const quaternion& b, double t) {
		double dot = quaternion::dot(a, b);
		if (dot < 0) {
			a = -a;
			dot = -dot;
		}
		
		const double DOT_THRESHOLD = 0.995;
		if (dot > DOT_THRESHOLD) {
			quaternion out = a + (b-a)*t;
			out = out.normalize();
			return out;
		}
		
		double theta_0 = acos(dot);
		double theta = theta_0 * t;
		
		double sin_theta = sin(theta);
		double sin_theta_0 = sin(theta_0);
		
		double s0 = cos(theta) - dot * sin_theta / sin_theta_0;
		double s1 = sin_theta / sin_theta_0;
		
		return (a*s0) + (b*s1);
	}
}
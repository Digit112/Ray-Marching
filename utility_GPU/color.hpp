namespace util {
	class rgb;
	class hsv;

	class rgb {
	public:
		double r;       // a fraction between 0 and 1
		double g;       // a fraction between 0 and 1
		double b;       // a fraction between 0 and 1
		
		__device__ rgb() {}
		__device__ __host__ rgb(double r, double g, double b) : r(r), g(g), b(b) {}
		__device__ __host__ rgb(hsv in);
	};

	class hsv {
	public:
		double h;       // angle in degrees
		double s;       // a fraction between 0 and 1
		double v;       // a fraction between 0 and 1
		
		__device__ hsv() {}
		__device__ __host__ hsv(double h, double s, double v) : h(h), s(s), v(v) {}
		__device__ __host__ hsv(rgb in);
	};

	hsv::hsv(rgb in) {
		double      min, max, delta;

		min = in.r < in.g ? in.r : in.g;
		min = min  < in.b ? min  : in.b;

		max = in.r > in.g ? in.r : in.g;
		max = max  > in.b ? max  : in.b;

		v = max;                                // v
		delta = max - min;
		if (delta < 0.00001)
		{
			s = 0;
			h = 0; // undefined, maybe nan?
			return;
		}
		if( max > 0.0 ) { // NOTE: if Max is == 0, this divide would cause a crash
			s = (delta / max);                  // s
		} else {
			// if max is 0, then r = g = b = 0              
			// s = 0, h is undefined
			s = 0.0;
			h = NAN;                            // its now undefined
			return;
		}
		if( in.r >= max )                           // > is bogus, just keeps compilor happy
			h = ( in.g - in.b ) / delta;        // between yellow & magenta
		else
		if( in.g >= max )
			h = 2.0 + ( in.b - in.r ) / delta;  // between cyan & yellow
		else
			h = 4.0 + ( in.r - in.g ) / delta;  // between magenta & cyan

		h *= 60.0;                              // degrees

		if( h < 0.0 )
			h += 360.0;
	}

	rgb::rgb(hsv in) {
		double      hh, p, q, t, ff;
		long        i;

		if(in.s <= 0.0) {       // < is bogus, just shuts up warnings
			r = in.v;
			g = in.v;
			b = in.v;
			return;
		}
		hh = in.h;
		if(hh >= 360.0) hh = 0.0;
		hh /= 60.0;
		i = (long)hh;
		ff = hh - i;
		p = in.v * (1.0 - in.s);
		q = in.v * (1.0 - (in.s * ff));
		t = in.v * (1.0 - (in.s * (1.0 - ff)));

		switch(i) {
		case 0:
			r = in.v;
			g = t;
			b = p;
			break;
		case 1:
			r = q;
			g = in.v;
			b = p;
			break;
		case 2:
			r = p;
			g = in.v;
			b = t;
			break;
		case 3:
			r = p;
			g = q;
			b = in.v;
			break;
		case 4:
			r = t;
			g = p;
			b = in.v;
			break;
		case 5:
		default:
			r = in.v;
			g = p;
			b = q;
			break;
		}
	}
}
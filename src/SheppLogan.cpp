/*
-----------------------------------------------------------------------
Copyright: 2010-2022, imec Vision Lab, University of Antwerp
           2014-2022, CWI, Amsterdam

Contact: astra@astra-toolbox.com
Website: http://www.astra-toolbox.com/

This file is part of the ASTRA Toolbox.


The ASTRA Toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The ASTRA Toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------------------------
*/

#include "astra/SheppLogan.h"
#include "astra/Float32VolumeData2D.h"
#include "astra/Data3D.h"

namespace astra {

struct Ellipse {
	double x;
	double y;
	double axisx;
	double axisy;
	double rotation;
	double value;
};

struct Ellipsoid {
	double x;
	double y;
	double z;
	double axisx;
	double axisy;
	double axisz;
	double rotation;
	double value;
};


template<typename T>
static void add_ellipse(T *data, unsigned int width, unsigned int height, const Ellipse &ell)
{
	double a = (0.5 + 0.5 * ell.x) * width - 0.5;
	double b = (0.5 - 0.5 * ell.y) * height - 0.5;
	double c = 0.5 * ell.axisx * width;
	double d = 0.5 * ell.axisy * height;
	double th = ell.rotation * PI / 180.0;

	double costh = cos(th);
	double costh2 = costh * costh;
	double sinth = sin(th);
	double sinth2 = sinth * sinth;
	double a2 = a * a;
	double b2 = b * b;
	double c2 = c * c;
	double d2 = d * d;

	// Sage:
	// var('x y a b c d th')
	// f = ((x - a) * cos(th) - (y - b) * sin(th))^2 / c^2 + ((x - a)*sin(th) + (y - b) * cos(th))^2/d^2 - 1
	// L = [ (s.coefficients(y),t) for s,t in f.expand().coefficients(x) ]

	double c20 = costh2/c2 + sinth2/d2;
	double c10 = -2*a*costh2/c2 + 2*b*costh*sinth/c2 - 2*b*costh*sinth/d2 - 2*a*sinth2/d2;
	double c11 = -2*costh*sinth/c2 + 2*costh*sinth/d2;
	double c00 = a2*costh2/c2 + b2*costh2/d2 - 2*a*b*costh*sinth/c2 + 2*a*b*costh*sinth/d2 + b2*sinth2/c2 + a2*sinth2/d2 - 1;
	double c01 = -2*b*costh2/d2 + 2*a*costh*sinth/c2 - 2*a*costh*sinth/d2 - 2*b*sinth2/c2;
	double c02 = costh2/d2 + sinth2/c2;

	for (unsigned int y = 0; y < height; ++y) {
		double A = c20;
		double B = y*c11 + c10;
		double C = y*y*c02 + y*c01 + c00;
		double D = B*B - 4*A*C;
		if (D < 0)
			continue;

		unsigned int xmin = ceil(0.5 * (-B - sqrt(D)) / A);
		unsigned int xmax = floor(0.5 * (-B + sqrt(D)) / A) + 1;
		if (xmin < 0) xmin = 0;
		if (xmax > width) xmax = width;

		if (xmin >= width)
			continue;
		if (xmax <= 0)
			continue;

		assert(xmax >= xmin);

		T *p = data + y * width + xmin;
		for (unsigned int x = 0; x < xmax - xmin; ++x)
			*p++ += (T)ell.value;
	}
}

template<typename T>
static void add_ellipsoid(T *data, unsigned int width, unsigned int height, unsigned int depth, const Ellipsoid &ell)
{
	double a = (0.5 + 0.5 * ell.x) * width - 0.5;
	double b = (0.5 - 0.5 * ell.y) * height - 0.5;
	double c = 0.5 * ell.axisx * width;
	double d = 0.5 * ell.axisy * height;
	double e = (0.5 - 0.5 * ell.z) * depth - 0.5;
	double f = 0.5 * ell.axisz * depth;
	double th = ell.rotation * PI / 180.0;

	double costh = cos(th);
	double costh2 = costh * costh;
	double sinth = sin(th);
	double sinth2 = sinth * sinth;
	double a2 = a * a;
	double b2 = b * b;
	double c2 = c * c;
	double d2 = d * d;

	// Sage:
	// var('x y a b c d th')
	// f = ((x - a) * cos(th) - (y - b) * sin(th))^2 / c^2 + ((x - a)*sin(th) + (y - b) * cos(th))^2/d^2 - 1
	// L = [ (s.coefficients(y),t) for s,t in f.expand().coefficients(x) ]

	double c20 = costh2/c2 + sinth2/d2;
	double c10 = -2*a*costh2/c2 + 2*b*costh*sinth/c2 - 2*b*costh*sinth/d2 - 2*a*sinth2/d2;
	double c11 = -2*costh*sinth/c2 + 2*costh*sinth/d2;
	double c00 = a2*costh2/c2 + b2*costh2/d2 - 2*a*b*costh*sinth/c2 + 2*a*b*costh*sinth/d2 + b2*sinth2/c2 + a2*sinth2/d2 - 1;
	double c01 = -2*b*costh2/d2 + 2*a*costh*sinth/c2 - 2*a*costh*sinth/d2 - 2*b*sinth2/c2;
	double c02 = costh2/d2 + sinth2/c2;

	for (unsigned int y = 0; y < height; ++y) {
		double A = c20;
		double B = y*c11 + c10;
		double C = y*y*c02 + y*c01 + c00;
		double D = B*B - 4*A*C;
		if (D < 0)
			continue;

		unsigned int xmin = ceil(0.5 * (-B - sqrt(D)) / A);
		unsigned int xmax = floor(0.5 * (-B + sqrt(D)) / A) + 1;
		if (xmin < 0) xmin = 0;
		if (xmax > width) xmax = width;

		if (xmin >= width)
			continue;
		if (xmax <= 0)
			continue;

		assert(xmax >= xmin);

		for (unsigned int x = 0; x < xmax - xmin; ++x) {
			double xy = A*(x + xmin)*(x + xmin) + B*(x + xmin) + C;
			unsigned int zmin = ceil(e - f * sqrt(-xy));
			unsigned int zmax = floor(e + f * sqrt(-xy)) + 1;
			if (zmin < 0) zmin = 0;
			if (zmax > depth) zmax = depth;

			if (zmin >= depth)
				continue;
			if (zmax <= 0)
				continue;

			assert(zmax >= zmin);

			T *p = data + (zmin * height + y) * width + (xmin + x);
			for (unsigned int z = 0; z < zmax - zmin; ++z) {
				*p += (T)ell.value;
				p += width * height;
			}
		}
	}
}


void generateSheppLogan(CFloat32VolumeData2D *data, bool modified) {
   	std::vector<Ellipse> ells = {
	//x,    y,      axisx,   axisy,rot, value
 	{ 0,    0,      0.69,   0.92,   0,  2 },
 	{ 0,   -0.0184, 0.6624, 0.874,  0, -0.98 },
 	{ 0.22, 0,      0.11,   0.31, -18, -0.02 },
 	{-0.22, 0,      0.16,   0.41,  18, -0.02 },
 	{ 0,    0.35,   0.21,   0.25,   0,  0.01 },
 	{ 0,    0.1,    0.046,  0.046,  0,  0.01 },
 	{ 0,   -0.1,    0.046,  0.046,  0,  0.01 },
 	{-0.08,-0.605,  0.046,  0.023,  0,  0.01 },
 	{ 0,   -0.605,  0.023,  0.023,  0,  0.01 },
 	{ 0.06,-0.605,  0.023,  0.046,  0,  0.01 } };

	std::vector<double> modvalues = { 1.0, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };

	if (modified) {
		assert(ells.size() == modvalues.size());
		for (unsigned int i = 0; i < ells.size(); ++i)
			ells[i].value = modvalues[i];
	}

	float32 *ptr = data->getData();

	for (size_t i = 0; i < data->getSize(); ++i)
		ptr[i] = 0.0f;

	for (const Ellipse &ell: ells)
		add_ellipse(ptr, data->getWidth(), data->getHeight(), ell);

	for (size_t i = 0; i < data->getSize(); ++i)
		if (ptr[i] < 0)
			ptr[i] = 0.0f;
}

void generateSheppLogan3D(CFloat32VolumeData3D *data, bool modified) {
	ASTRA_ASSERT(data->isFloat32Memory());

	std::vector<Ellipsoid> ells = {
	//x,    y,      z, axisx,  axisy,axisz,rot,value
	{ 0,    0,      0, 0.69,   0.92,  0.81,  0, 2.00 },
	{ 0,   -0.0184, 0, 0.6624, 0.874, 0.78,  0,-0.98 },
	{ 0.22, 0,      0, 0.11,   0.31,  0.22,-18,-0.02 },
	{-0.22, 0,      0, 0.16,   0.41,  0.28, 18,-0.02 },
	{ 0,    0.35,   0, 0.21,   0.25,  0.41,  0, 0.01 },
	{ 0,    0.1,    0, 0.046,  0.046, 0.05,  0, 0.01 },
	{ 0,   -0.1,    0, 0.046,  0.046, 0.05,  0, 0.01 },
	{-0.08,-0.605,  0, 0.046,  0.023, 0.05,  0, 0.01 },
	{ 0,   -0.605,  0, 0.023,  0.023, 0.02,  0, 0.01 },
	{ 0.06,-0.605,  0, 0.023,  0.046, 0.02,  0, 0.01 } };

	std::vector<double> modvalues = { 1.0, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };

	if (modified) {
		assert(ells.size() == modvalues.size());
		for (unsigned int i = 0; i < ells.size(); ++i)
			ells[i].value = modvalues[i];
	}

	float32 *ptr = data->getFloat32Memory();

	for (size_t i = 0; i < data->getSize(); ++i)
		ptr[i] = 0.0f;

	for (const Ellipsoid &ell: ells)
		add_ellipsoid(ptr, data->getWidth(), data->getHeight(), data->getDepth(), ell);

	for (size_t i = 0; i < data->getSize(); ++i)
		if (ptr[i] < 0)
			ptr[i] = 0.0f;
}

}

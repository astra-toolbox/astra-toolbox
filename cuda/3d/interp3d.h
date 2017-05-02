/*
-----------------------------------------------------------------------
Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
           2014-2016, CWI, Amsterdam

Contact: astra@uantwerpen.be
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

#ifndef _CUDA_INTERP3D_H
#define _CUDA_INTERP3D_H

#include <cuda.h>

namespace astraCUDA3d {

// interpolation routines
__device__
float tex_interpolate(float f0, float f1, float f2, float (*evaluate)(float,float,float)) {
        return evaluate(f0, f1, f2);
}


__device__
float bilin_interpolate(float f0, float f1, float f2, float (*evaluate)(float,float,float)) {
        float f1_lower = floorf(f1 - 0.5f) + 0.5f;
        float df1 = f1-f1_lower;
        float f2_lower = floorf(f2 - 0.5f) + 0.5f;
        float df2 = f2-f2_lower;
        return   (1.0f-df2) * (   (1.0f-df1) * evaluate(f0, f1_lower       , f2_lower       ) 
                                +       df1  * evaluate(f0, f1_lower + 1.0f, f2_lower       ) )
               +       df2  * (   (1.0f-df1) * evaluate(f0, f1_lower       , f2_lower + 1.0f) 
                                +       df1  * evaluate(f0, f1_lower + 1.0f, f2_lower + 1.0f) );
}


template<float (*interp_kernel_f1)(float), float (*interp_kernel_f2)(float)>
__device__
inline float bicubic_interpolate(float f0, float f1, float f2, float (*evaluate)(float,float,float)) {

        float f1_lowest = floorf(f1 - 0.5f) - 0.5f;
        float f2_lowest = floorf(f2 - 0.5f) - 0.5f;

        float f1_rel_to_lowest = f1 - f1_lowest;
        float f2_rel_to_lowest = f2 - f2_lowest;

        float w1[4], w2[4];
        for(int i = 0; i < 4; i++) {
                w1[i] = interp_kernel_f1(f1_rel_to_lowest - i);
                w2[i] = interp_kernel_f2(f2_rel_to_lowest - i);
        }

        float result = 0.0f;
        for(int step1 = 0; step1 < 4; step1++) {
                for(int step2 = 0; step2 < 4; step2++) {
                        result += w1[step1] * w2[step2] * evaluate(f0, f1_lowest + step1, f2_lowest + step2);
                }
        }

        return result;

}


__device__
float cubic_hermite_spline_eval(float x) {
        x = fabs(x);
        if (x >= 2.0f)
                return 0.0f;

        float x2 = x*x;
        if (x <= 1.0f)
                return (1.5f * x - 2.5f) * x2 + 1.0f;

        return (-0.5f * x + 2.5f) * x2 - 4.0f * x + 2.0f;
}


__device__
float cubic_hermite_spline_deriv(float x) {
        float abs_x = fabs(x);
        if (abs_x >= 2.0f)
                return 0.0f;

        float sgn_x = copysignf(1.0f, x);
        if (abs_x <= 1.0f)
                return sgn_x * ((4.5f * abs_x - 5.0f) * abs_x);

        return sgn_x * ((-1.5f * abs_x + 5.0f) * abs_x - 4.0f);
}


__device__
float bicubic_interpolate(float f0, float f1, float f2, float (*evaluate)(float,float,float)) {
        return bicubic_interpolate<cubic_hermite_spline_eval, cubic_hermite_spline_eval>(f0, f1, f2, evaluate);
}


__device__
float bicubic_interpolate_ddf1(float f0, float f1, float f2, float (*evaluate)(float,float,float)) {
        return bicubic_interpolate<cubic_hermite_spline_deriv, cubic_hermite_spline_eval>(f0, f1, f2, evaluate);
}


__device__
float bicubic_interpolate_ddf2(float f0, float f1, float f2, float (*evaluate)(float,float,float)) {
        return bicubic_interpolate<cubic_hermite_spline_eval, cubic_hermite_spline_deriv>(f0, f1, f2, evaluate);
}

}

#endif

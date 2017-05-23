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



__device__
inline float pow2(float x) { return x*x; }

__device__
inline float pow3(float x) { return x*x*x; }



__device__
inline float tex_interpolate(float f0, float f1, float f2, float (*texture_lookup)(float,float,float)) {
        return texture_lookup(f0, f1, f2);
}


__device__
inline float bilin_interpolate(float f0, float f1, float f2, float (*texture_lookup)(float,float,float)) {
        float f1_lower = floorf(f1 - 0.5f) + 0.5f;
        float df1 = f1-f1_lower;
        float f2_lower = floorf(f2 - 0.5f) + 0.5f;
        float df2 = f2-f2_lower;
        return   (1.0f-df2) * (   (1.0f-df1) * texture_lookup(f0, f1_lower       , f2_lower       ) 
                                +       df1  * texture_lookup(f0, f1_lower + 1.0f, f2_lower       ) )
               +       df2  * (   (1.0f-df1) * texture_lookup(f0, f1_lower       , f2_lower + 1.0f) 
                                +       df1  * texture_lookup(f0, f1_lower + 1.0f, f2_lower + 1.0f) );
}


template<float (*interp_kernel_f1)(float), float (*interp_kernel_f2)(float)>
__device__
inline float bicubic_interpolate(float f0, float f1, float f2, float (*texture_lookup)(float,float,float)) {

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
                        result += w1[step1] * w2[step2] * texture_lookup(f0, f1_lowest + step1, f2_lowest + step2);
                }
        }

        return result;

}



template<void (*get_bilin_coeffs_f1)(float, bool, float &, float &), void (*get_bilin_coeffs_f2)(float, bool, float &, float &)>
__device__
inline float bspline3_interpolate(float f0, float f1, float f2, float (*texture_lookup)(float,float,float)) {

        float f1_lower = floorf(f1 - 0.5f) + 0.5f;
        float df1 = f1-f1_lower;
        float f2_lower = floorf(f2 - 0.5f) + 0.5f;
        float df2 = f2-f2_lower;

        // Compute auxillary points for bilinear interpolations and
        // corresponding weights
        float y1_plus, w1_plus, y1_minus, w1_minus, y2_plus, w2_plus, y2_minus, w2_minus;
        get_bilin_coeffs_f1(df1, false, y1_plus, w1_plus);
        get_bilin_coeffs_f1(1.0f - df1, true, y1_minus, w1_minus);
        get_bilin_coeffs_f2(df2, false, y2_plus, w2_plus);
        get_bilin_coeffs_f2(1.0f - df2, true, y2_minus, w2_minus);

        // Shift auxillary interpolation points to actually lie between the
        // correct grid points
        y1_plus += f1_lower;
        y2_plus += f2_lower;
        y1_minus += f1_lower;
        y2_minus += f2_lower;

        return   w1_plus  * w2_plus  * texture_lookup(f0, y1_plus,  y2_plus )
               + w1_minus * w2_plus  * texture_lookup(f0, y1_minus, y2_plus )
               + w1_plus  * w2_minus * texture_lookup(f0, y1_plus,  y2_minus)
               + w1_minus * w2_minus * texture_lookup(f0, y1_minus, y2_minus);

}



__device__
inline void get_bilin_coeffs_b3_eval(float x, bool is_left, float & y, float & w) {
        float x_plus_1_cube = pow3(x + 1.0f);
        float x_cube = pow3(x);
        w = x_plus_1_cube - 3.0f * x_cube;
        y = x_cube / w;
        w = 0.1666666f * w;
        if(is_left) {
                y = -y;
        } else {
                y += 1.0f;
        }
}



__device__
inline void get_bilin_coeffs_b3_deriv(float x, bool is_left, float & y, float & w) {
        float x_plus_1_sq = pow2(x + 1.0f);
        float x_sq = pow2(x);
        w = x_plus_1_sq - 3.0f * x_sq;
        y = x_sq / w;
        w = 0.5f * w;
        if(is_left) {
                y = -y;
                w = -w;
        } else {
                y += 1.0f;
        }
}




__device__
inline float cubic_hermite_spline_eval(float x) {
        x = fabs(x);
        if (x >= 2.0f)
                return 0.0f;

        float x2 = x*x;
        if (x <= 1.0f)
                return (1.5f * x - 2.5f) * x2 + 1.0f;

        return (-0.5f * x + 2.5f) * x2 - 4.0f * x + 2.0f;
}


__device__
inline float cubic_hermite_spline_deriv(float x) {
        float abs_x = fabs(x);
        if (abs_x >= 2.0f)
                return 0.0f;

        float sgn_x = copysignf(1.0f, x);
        if (abs_x <= 1.0f)
                return sgn_x * ((4.5f * abs_x - 5.0f) * abs_x);

        return sgn_x * ((-1.5f * abs_x + 5.0f) * abs_x - 4.0f);
}




__device__
inline float b3_spline_eval(float x) {
        x = fabs(x);
        if (x >= 2.0f)
                return 0.0f;

        float res = pow3(2.0f-x);
        if (x <= 1.0f)
                res -= 4.0f*pow3(1.0f-x);

        return 0.166666f * res;
}


__device__
inline float b3_spline_deriv(float x) {
        float sgn_x = copysignf(1.0f, x);
        x = fabs(x);
        if (x >= 2.0f)
                return 0.0f;
        
        float res = pow2(2.0f-x);
        if (x <= 1.0f)
                res -= 4.0f*pow2(1.0f-x);

        return -sgn_x * 0.5f * res;
}


__device__
inline float bicubic_interpolate(float f0, float f1, float f2, float (*texture_lookup)(float,float,float)) {
        return bicubic_interpolate<cubic_hermite_spline_eval, cubic_hermite_spline_eval>(f0, f1, f2, texture_lookup);
        //return bspline3_interpolate<get_bilin_coeffs_b3_eval, get_bilin_coeffs_b3_eval>(f0, f1, f2, texture_lookup);
        //return bicubic_interpolate<b3_spline_eval, b3_spline_eval>(f0, f1, f2, texture_lookup);
}


__device__
inline float bicubic_interpolate_ddf1(float f0, float f1, float f2, float (*texture_lookup)(float,float,float)) {
        return bicubic_interpolate<cubic_hermite_spline_deriv, cubic_hermite_spline_eval>(f0, f1, f2, texture_lookup);
        //return bspline3_interpolate<get_bilin_coeffs_b3_deriv, get_bilin_coeffs_b3_eval>(f0, f1, f2, texture_lookup);
        //return bicubic_interpolate<b3_spline_deriv, b3_spline_eval>(f0, f1, f2, texture_lookup);
}


__device__
inline float bicubic_interpolate_ddf2(float f0, float f1, float f2, float (*texture_lookup)(float,float,float)) {
        return bicubic_interpolate<cubic_hermite_spline_eval, cubic_hermite_spline_deriv>(f0, f1, f2, texture_lookup);
        //return bspline3_interpolate<get_bilin_coeffs_b3_eval, get_bilin_coeffs_b3_deriv>(f0, f1, f2, texture_lookup);
        //return bicubic_interpolate<b3_spline_eval, b3_spline_deriv>(f0, f1, f2, texture_lookup);
}


}

#endif

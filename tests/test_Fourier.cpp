/*
-----------------------------------------------------------------------
Copyright 2012 iMinds-Vision Lab, University of Antwerp

Contact: astra@ua.ac.be
Website: http://astra.ua.ac.be


This file is part of the
All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA Toolbox").

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
$Id$
*/


#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/auto_unit_test.hpp>
#include <boost/test/test_case_template.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "astra/Fourier.h"

BOOST_AUTO_TEST_CASE( testFourier_DFT_1D_1 )
{
	astra::float32 inR[5] = { 1.0f, 1.0f, 0.0f, 0.0f, 1.0f };
	astra::float32 inI[5] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	astra::float32 outR[5];
	astra::float32 outI[5];

	astra::discreteFourierTransform1D(5, inR, inI, outR, outI, 1, 1, false);

	astra::float32 expected1R[5] = { 3.0f, 1.618034f, -0.618034f, -0.618034f, 1.618034f };
	for (unsigned int i = 0; i < 5; ++i) {
		BOOST_CHECK_SMALL(outR[i] - expected1R[i], 0.00001f);
		BOOST_CHECK_SMALL(outI[i], 0.00001f);
	}

	astra::discreteFourierTransform1D(5, outR, outI, inR, inI, 1, 1, true);
	astra::float32 expected2R[5] = { 1.0f, 1.0f, 0.0f, 0.0f, 1.0f };
	for (unsigned int i = 0; i < 5; ++i) {
		BOOST_CHECK_SMALL(inR[i] - expected2R[i], 0.00001f);
		BOOST_CHECK_SMALL(inI[i], 0.00001f);
	}
}

BOOST_AUTO_TEST_CASE( testFourier_DFT_2D_1 )
{
	astra::float32 inR[25] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
	                           1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
	                           1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                           1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                           1.0f, 1.0f, 0.0f, 0.0f, 1.0f };
	astra::float32 inI[25] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	astra::float32 outR[25];
	astra::float32 outI[25];

	astra::discreteFourierTransform2D(5, 5, inR, inI, outR, outI, false);

	astra::float32 expected1R[25] =
	     { 13.0f    , 5.236068f, 0.763932f, 0.763932f, 5.236068f,
	       5.236068f,-0.618034f,-2.0f     ,-2.0f     ,-0.618034f,
	       0.763932f,-2.0f     , 1.618034f, 1.618034f,-2.0f     ,
	       0.763932f,-2.0f     , 1.618034f, 1.618034f,-2.0f     ,
	       5.236068f,-0.618034f,-2.0f     ,-2.0f     ,-0.618034f };
	for (unsigned int i = 0; i < 25; ++i) {
		BOOST_CHECK_SMALL(outR[i] - expected1R[i], 0.00001f);
		BOOST_CHECK_SMALL(outI[i], 0.00001f);
	}

	astra::discreteFourierTransform2D(5, 5, outR, outI, inR, inI, true);
	astra::float32 expected2R[25] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
	                                  1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
	                                  1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                                  1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                                  1.0f, 1.0f, 0.0f, 0.0f, 1.0f };
	for (unsigned int i = 0; i < 25; ++i) {
		BOOST_CHECK_SMALL(inR[i] - expected2R[i], 0.00001f);
		BOOST_CHECK_SMALL(inI[i], 0.00001f);
	}


}


BOOST_AUTO_TEST_CASE( testFourier_FFT_1D_1 )
{
	astra::float32 inR[8] = { 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f };
	astra::float32 inI[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	astra::float32 outR[8];
	astra::float32 outI[8];

	astra::fastTwoPowerFourierTransform1D(8, inR, inI, outR, outI, 1, 1, false);

	astra::float32 expected1R[8] = { 5.0f, 2.414214f, -1.0f, -0.414214f, 1.0f, -0.414214f, -1.0f, 2.414214f };
	for (unsigned int i = 0; i < 8; ++i) {
		BOOST_CHECK_SMALL(outR[i] - expected1R[i], 0.00001f);
		BOOST_CHECK_SMALL(outI[i], 0.00001f);
	}

	astra::fastTwoPowerFourierTransform1D(8, outR, outI, inR, inI, 1, 1, true);
	astra::float32 expected2R[8] = { 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f };
	for (unsigned int i = 0; i < 8; ++i) {
		BOOST_CHECK_SMALL(inR[i] - expected2R[i], 0.00001f);
		BOOST_CHECK_SMALL(inI[i], 0.00001f);
	}

}

BOOST_AUTO_TEST_CASE( testFourier_FFT_2D_1 )
{
	astra::float32 inR[64] = { 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f,
	                           1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
	                           1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
	                           1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                           1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                           1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
	                           1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f };
	astra::float32 inI[64] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                           0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	astra::float32 outR[64];
	astra::float32 outI[64];

	astra::discreteFourierTransform2D(8, 8, inR, inI, outR, outI, false);

	astra::float32 expected1R[64] =
	     { 25.0f, 12.656854f, 1.0f, 1.343146f, 1.0f, 1.343146f, 1.0f, 12.656854f,
	       12.656854f, 3.0f, -3.828427f, -1.0f, -1.0f, -1.0f, -3.828427f, 3.0f,
	       1.0f, -3.828427f, -3.0f, 1.828427f, 1.0f, 1.828427f, -3.0f, -3.828427f,
	       1.343146f, -1.0f, 1.828427f, 3.0f, -1.0f, 3.0f, 1.828427f, -1.0f,
	       1.0f, -1.0f, 1.0f, -1.0f, -7.0f, -1.0f, 1.0f, -1.0f,
	       1.343146f, -1.0f, 1.828427f, 3.0f, -1.0f, 3.0f, 1.828427f, -1.0f,
	       1.0f, -3.828427f, -3.0f, 1.828427f, 1.0f, 1.828427f, -3.0f, -3.828427f,
	       12.656854f, 3.0f, -3.828427f, -1.0f, -1.0f, -1.0f, -3.828427f, 3.0f };
	for (unsigned int i = 0; i < 64; ++i) {
		BOOST_CHECK_SMALL(outR[i] - expected1R[i], 0.00002f);
		BOOST_CHECK_SMALL(outI[i], 0.00001f);
	}


	astra::discreteFourierTransform2D(8, 8, outR, outI, inR, inI, true);
	astra::float32 expected2R[64] = { 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f,
	                                  1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
	                                  1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
	                                  1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                                  1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	                                  1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
	                                  1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f };
	for (unsigned int i = 0; i < 64; ++i) {
		BOOST_CHECK_SMALL(inR[i] - expected2R[i], 0.00001f);
		BOOST_CHECK_SMALL(inI[i], 0.00001f);
	}


}


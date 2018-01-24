/*
-----------------------------------------------------------------------
Copyright: 2010-2018, imec Vision Lab, University of Antwerp
           2014-2018, CWI, Amsterdam

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


#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/auto_unit_test.hpp>
#include <boost/test/test_case_template.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include "astra/Fourier.h"

BOOST_AUTO_TEST_CASE( testFourier_FFT_1D_1 )
{
	astra::float32 data[16] = { 1.0f,0.0f, 1.0f,0.0f, 1.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 1.0f,0.0f, 1.0f,0.0f };
	int ip[6];
	astra::float32 w[8];
	ip[0] = 0;

	astra::cdft(16, -1, data, ip, w);

	astra::float32 expected1[16] = { 5.0f,0.0f, 2.414214f,0.0f, -1.0f,0.0f, -0.414214f,0.0f, 1.0f,0.0f, -0.414214f,0.0f, -1.0f,0.0f, 2.414214f,0.0f };
	for (unsigned int i = 0; i < 16; ++i) {
		BOOST_CHECK_SMALL(data[i] - expected1[i], 0.00001f);
	}

	astra::cdft(16, 1, data, ip, w);
	astra::float32 expected2[16] = { 8.0f,0.0f, 8.0f,0.0f, 8.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 0.0f,0.0f, 8.0f,0.0f, 8.0f,0.0f };
	for (unsigned int i = 0; i < 16; ++i) {
		BOOST_CHECK_SMALL(data[i] - expected2[i], 0.00001f);
	}

}


/*
-----------------------------------------------------------------------
Copyright: 2010-2015, iMinds-Vision Lab, University of Antwerp
           2014-2015, CWI, Amsterdam

Contact: astra@uantwerpen.be
Website: http://sf.net/projects/astra-toolbox

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
$Id$
*/

#ifndef FBP_FILTERS_H
#define FBP_FILTERS_H

enum E_FBPFILTER
{
	FILTER_NONE,			//< no filter (regular BP)
	FILTER_RAMLAK,			//< default FBP filter
	FILTER_SHEPPLOGAN,		//< Shepp-Logan
	FILTER_COSINE,			//< Cosine
	FILTER_HAMMING,			//< Hamming filter
	FILTER_HANN,			//< Hann filter
	FILTER_TUKEY,			//< Tukey filter
	FILTER_LANCZOS,			//< Lanczos filter
	FILTER_TRIANGULAR,		//< Triangular filter
	FILTER_GAUSSIAN,		//< Gaussian filter
	FILTER_BARTLETTHANN,	//< Bartlett-Hann filter
	FILTER_BLACKMAN,		//< Blackman filter
	FILTER_NUTTALL,			//< Nuttall filter, continuous first derivative
	FILTER_BLACKMANHARRIS,	//< Blackman-Harris filter
	FILTER_BLACKMANNUTTALL,	//< Blackman-Nuttall filter
	FILTER_FLATTOP,			//< Flat top filter
	FILTER_KAISER,			//< Kaiser filter
	FILTER_PARZEN,			//< Parzen filter
	FILTER_PROJECTION,		//< all projection directions share one filter
	FILTER_SINOGRAM,		//< every projection direction has its own filter
	FILTER_RPROJECTION,		//< projection filter in real space (as opposed to fourier space)
	FILTER_RSINOGRAM,		//< sinogram filter in real space
};

#endif /* FBP_FILTERS_H */

# -----------------------------------------------------------------------
# Copyright: 2010-2016, iMinds-Vision Lab, University of Antwerp
#            2013-2016, CWI, Amsterdam
#
# Contact: astra@uantwerpen.be
# Website: http://sf.net/projects/astra-toolbox
#
# This file is part of the ASTRA Toolbox.
#
#
# The ASTRA Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The ASTRA Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------
# distutils: language = c++


def clipCircle(img):
	cdef int i,j
	cdef double x2,y2,mid,bnd
	cdef long sz,sz2
	sz = img.shape[0]
	sz2 = sz*sz
	bnd = sz2/4.
	mid = (sz-1.)/2.
	nDel=0
	for i in range(sz):
		for j in range(sz):
			x2 = (i-mid)*(i-mid)
			y2 = (j-mid)*(j-mid)
			if x2+y2>bnd:
				img[i,j]=0
				nDel=nDel+1
	return nDel

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


#ifndef _ASTRA_CUDA_STREAM_H
#define _ASTRA_CUDA_STREAM_H

#include <memory>

namespace astraCUDA {

// Internal structure hiding CUDA types
struct Stream_internal;

// type tag for differentiating between the Stream constructors
struct automatic_sync_t { };
constexpr automatic_sync_t automatic_sync {};



// Utility class wrapping a CUDA stream for use in astra's C++ code

class Stream {
public:
	// Create new stream
	Stream();

	// Create new stream that is marked as needing automatic syncs.
	// Intended for use in temporary objects for default arguments to utility functions.
	Stream(automatic_sync_t);

	// Wrap around existing stream.
	// Note that Stream_internal has a (non-explicit) constructor taking a cudaStream_t argument
	Stream(Stream_internal stream);

	// Destroys the stream if created by this class
	~Stream();

	// Check if stream creation in the constructor was successful
	bool isValid() const;

	// Synchronize on this stream
	// 'msg' is added to the printed error message if this causes or reports a cuda error
	bool sync(const char *msg) const;

	// Synchronize on this stream if this is an auto-sync stream.
	bool syncIfAuto(const char *msg) const;

	Stream_internal &operator*();
	const Stream_internal &operator*() const;

private:
	std::unique_ptr<Stream_internal> impl;

};

}

#endif

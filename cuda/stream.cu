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

#include "astra/cuda/gpu_runtime_wrapper.h"

#include "astra/cuda/stream.h"
#include "astra/cuda/stream_internal.h"
#include "astra/cuda/2d/util.h"

namespace astraCUDA {

Stream::Stream()
{
	cudaStream_t stream;
	if (!checkCuda(cudaStreamCreate(&stream), "CUDAStreamWrapper"))
		return;

	impl = std::make_unique<Stream_internal>(stream);
	impl->owned = true; // destroy this stream in our destructor
}

Stream::Stream(automatic_sync_t) : Stream()
{
	if (isValid())
		impl->autoSync = true;
}

Stream::Stream(Stream_internal s)
{
	impl = std::make_unique<Stream_internal>(s);
}

Stream::~Stream()
{
	if (isValid()) {
		if (impl->autoSync)
			logCuda(cudaStreamSynchronize(impl->stream), "Stream autoSync");

		if (impl->owned)
			logCuda(cudaStreamDestroy(impl->stream), "Stream destroy");
	}
}

bool Stream::isValid() const
{
	return impl.get() != nullptr;
}

bool Stream::sync(const char *msg) const
{
	if (!checkCuda(cudaStreamSynchronize(impl->stream), msg))
		return false;
	return true;
}

bool Stream::syncIfAuto(const char *msg) const
{
	if (impl->autoSync)
		return sync(msg);
	return true;
}


Stream_internal &Stream::operator*()
{
	return *impl.get();
}

const Stream_internal &Stream::operator*() const
{
	return *impl.get();
}



}

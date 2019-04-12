FROM debian:7 AS BUILDBASE
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y perl-modules build-essential autoconf libtool automake libboost-dev git && rm -rf /var/lib/apt/lists/*

FROM BUILDBASE AS CUDA100
RUN touch /root/cuda100
COPY cuda_10.0.130_410.48_linux /root
RUN /bin/bash /root/cuda_10.0.130_410.48_linux --toolkit --silent && \
	rm -f /root/cuda_10.0.130_410.48_linux

FROM BUILDBASE AS CUDA92
RUN touch /root/cuda92
COPY cuda_9.2.148_396.37_linux /root
RUN /bin/bash /root/cuda_9.2.148_396.37_linux --toolkit --silent && \
	rm -f /root/cuda_9.2.148_396.37_linux

FROM BUILDBASE AS CUDA90
RUN touch /root/cuda90
COPY cuda_9.0.176_384.81_linux-run /root
RUN /bin/bash /root/cuda_9.0.176_384.81_linux-run --toolkit --silent && \
	rm -f /root/cuda_9.0.176_384.81_linux-run

FROM BUILDBASE AS CUDA80
RUN touch /root/cuda80
COPY cuda_8.0.61_375.26_linux-run /root
RUN /bin/bash /root/cuda_8.0.61_375.26_linux-run --toolkit --silent && \
	rm -f /root/cuda_8.0.61_375.26_linux-run
COPY cuda_8.0.61.2_linux-run /root
RUN /bin/bash /root/cuda_8.0.61.2_linux-run --silent --accept-eula && \
	rm -f /root/cuda_8.0.61.2_linux-run

FROM BUILDBASE AS CUDA75
RUN touch /root/cuda75
COPY cuda_7.5.18_linux.run /root
RUN /bin/bash /root/cuda_7.5.18_linux.run --toolkit --silent && \
	rm -f /root/cuda_7.5.18_linux.run

FROM BUILDBASE AS CUDA70
RUN touch /root/cuda70
COPY cuda_7.0.28_linux.run /root
RUN /bin/bash /root/cuda_7.0.28_linux.run -toolkit -silent && \
	rm -f /root/cuda_7.0.28_linux.run

COPY cufft_patch_linux.tar.gz /root
RUN cd /usr/local/cuda-7.0 && \
	tar xf /root/cufft_patch_linux.tar.gz && \
	rm -f /root/cufft_patch_linux.tar.gz

FROM BUILDBASE AS CUDA60
RUN touch /root/cuda60
COPY cuda_6.0.37_linux_64.run /root
RUN /bin/bash /root/cuda_6.0.37_linux_64.run -toolkit -silent && \
	rm -f /root/cuda_6.0.37_linux_64.run

FROM BUILDBASE AS CUDA55
RUN touch /root/cuda55
COPY cuda_5.5.22_linux_64.run /root
RUN /bin/bash /root/cuda_5.5.22_linux_64.run -toolkit -silent && \
	rm /root/cuda_5.5.22_linux_64.run

FROM BUILDBASE
RUN touch /root/cuda
COPY --from=CUDA100 /usr/local/cuda-10.0 /usr/local/cuda-10.0
COPY --from=CUDA92 /usr/local/cuda-9.2 /usr/local/cuda-9.2
COPY --from=CUDA90 /usr/local/cuda-9.0 /usr/local/cuda-9.0
COPY --from=CUDA80 /usr/local/cuda-8.0 /usr/local/cuda-8.0
COPY --from=CUDA75 /usr/local/cuda-7.5 /usr/local/cuda-7.5
COPY --from=CUDA70 /usr/local/cuda-7.0 /usr/local/cuda-7.0
COPY --from=CUDA60 /usr/local/cuda-6.0 /usr/local/cuda-6.0
COPY --from=CUDA55 /usr/local/cuda-5.5 /usr/local/cuda-5.5

ENV PATH /root/miniconda3/bin:$PATH
COPY Miniconda3-4.5.4-Linux-x86_64.sh /root/
RUN /bin/bash /root/Miniconda3-4.5.4-Linux-x86_64.sh -b && \
	rm -f /root/Miniconda3*
RUN conda install -y conda-build conda-verify

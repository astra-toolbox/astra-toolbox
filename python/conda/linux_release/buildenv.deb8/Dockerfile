FROM debian:8 AS BUILDBASE
ENV DEBIAN_FRONTEND noninteractive
#RUN echo 'deb http://archive.debian.org/debian/ wheezy main' > /etc/apt/sources.list && echo 'deb http://archive.debian.org/debian-security/ wheezy/updates main' >> /etc/apt/sources.list && apt-get -o Acquire::Check-Valid-Until=false update && apt-get install -y perl-modules build-essential autoconf libtool automake libboost-dev git libxml2 && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y perl-modules build-essential autoconf libtool automake libboost-dev git libxml2 && rm -rf /var/lib/apt/lists/*

ENV PATH /root/miniconda3/bin:$PATH
COPY Miniconda3-py39_4.10.3-Linux-x86_64.sh /root/
RUN /bin/bash /root/Miniconda3-py39_4.10.3-Linux-x86_64.sh -b && \
	rm -f /root/Miniconda3*

RUN conda install -y conda-build conda-verify

# cudatoolkit < 9.0 is not easily available anymore from recent versions of conda.
# We have to enable the free channel (globally) to reenable old versions of cudatookit..
# See: https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/free-channel.html#troubleshooting
# TODO: Remove when support for cudatoolkit 8.0 is dropped.
RUN conda config --set restore_free_channel true

FROM BUILDBASE AS CUDA102
RUN touch /root/cuda102
COPY cuda_10.2.89_440.33.01_linux.run /root
RUN /bin/bash /root/cuda_10.2.89_440.33.01_linux.run --toolkit --silent --installpath=/usr/local/cuda-10.2 && \
	rm -f /root/cuda_10.2.89_440.33.01_linux.run


FROM BUILDBASE AS CUDA101
RUN touch /root/cuda101
COPY cuda_10.1.243_418.87.00_linux.run /root
RUN /bin/bash /root/cuda_10.1.243_418.87.00_linux.run --toolkit --silent --installpath=/usr/local/cuda-10.1 && \
	rm -f /root/cuda_10.1.243_418.87.00_linux.run

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

FROM BUILDBASE
RUN touch /root/cuda
COPY --from=CUDA102 /usr/local/cuda-10.2 /usr/local/cuda-10.2
COPY --from=CUDA101 /usr/local/cuda-10.1 /usr/local/cuda-10.1
COPY --from=CUDA100 /usr/local/cuda-10.0 /usr/local/cuda-10.0
COPY --from=CUDA92 /usr/local/cuda-9.2 /usr/local/cuda-9.2
COPY --from=CUDA90 /usr/local/cuda-9.0 /usr/local/cuda-9.0
COPY --from=CUDA80 /usr/local/cuda-8.0 /usr/local/cuda-8.0

RUN conda create -y -n prep -c nvidia --download-only cudatoolkit=10.2 && \
    conda create -y -n prep -c nvidia --download-only cudatoolkit=10.1 && \
    conda create -y -n prep -c nvidia --download-only cudatoolkit=10.0 && \
    conda create -y -n prep -c nvidia --download-only cudatoolkit=9.2 && \
    conda create -y -n prep -c nvidia --download-only cudatoolkit=9.0 && \
    conda create -y -n prep -c nvidia --download-only cudatoolkit=8.0

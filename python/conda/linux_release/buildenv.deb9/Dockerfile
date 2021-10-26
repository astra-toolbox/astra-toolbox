FROM debian:9 AS BUILDBASE
ENV DEBIAN_FRONTEND noninteractive
#RUN echo 'deb http://archive.debian.org/debian/ wheezy main' > /etc/apt/sources.list && echo 'deb http://archive.debian.org/debian-security/ wheezy/updates main' >> /etc/apt/sources.list && apt-get -o Acquire::Check-Valid-Until=false update && apt-get install -y perl-modules build-essential autoconf libtool automake libboost-dev git libxml2 && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y perl-modules build-essential autoconf libtool automake libboost-dev git libxml2 && rm -rf /var/lib/apt/lists/*

ENV PATH /root/miniconda3/bin:$PATH
COPY Miniconda3-py39_4.10.3-Linux-x86_64.sh /root/
RUN /bin/bash /root/Miniconda3-py39_4.10.3-Linux-x86_64.sh -b && \
	rm -f /root/Miniconda3*

RUN conda install -y conda-build conda-verify

FROM BUILDBASE AS CUDA111
RUN touch /root/cuda111
COPY cuda_11.1.1_455.32.00_linux.run /root
RUN /bin/bash /root/cuda_11.1.1_455.32.00_linux.run --toolkit --silent --installpath=/usr/local/cuda-11.1 && \
	rm -f /root/cuda_11.1.1_455.32.00_linux.run

FROM BUILDBASE AS CUDA110
RUN touch /root/cuda110
COPY cuda_11.0.3_450.51.06_linux.run /root
RUN /bin/bash /root/cuda_11.0.3_450.51.06_linux.run --toolkit --silent --installpath=/usr/local/cuda-11.0 && \
	rm -f /root/cuda_11.0.3_450.51.06_linux.run


FROM BUILDBASE
RUN touch /root/cuda
COPY --from=CUDA111 /usr/local/cuda-11.1 /usr/local/cuda-11.1
COPY --from=CUDA110 /usr/local/cuda-11.0 /usr/local/cuda-11.0

RUN conda create -y -n prep -c nvidia --download-only cudatoolkit=11.1 && \
    conda create -y -n prep -c nvidia --download-only cudatoolkit=11.0

FROM debian:8 AS BUILDBASE
ENV DEBIAN_FRONTEND noninteractive
#RUN echo 'deb http://archive.debian.org/debian/ wheezy main' > /etc/apt/sources.list && echo 'deb http://archive.debian.org/debian-security/ wheezy/updates main' >> /etc/apt/sources.list && apt-get -o Acquire::Check-Valid-Until=false update && apt-get install -y perl-modules build-essential autoconf libtool automake libboost-dev git libxml2 && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y perl-modules build-essential autoconf libtool automake libboost-dev git libxml2 && rm -rf /var/lib/apt/lists/*

ENV PATH /root/miniconda3/bin:$PATH
COPY Miniconda3-py38_4.8.3-Linux-x86_64.sh /root/
RUN /bin/bash /root/Miniconda3-py38_4.8.3-Linux-x86_64.sh -b && \
	rm -f /root/Miniconda3*

RUN conda install -y conda-build conda-verify

FROM BUILDBASE AS CUDA110
RUN touch /root/cuda110
COPY cuda_11.0.3_450.51.06_linux.run /root
RUN /bin/bash /root/cuda_11.0.3_450.51.06_linux.run --toolkit --silent --installpath=/usr/local/cuda-11.0 && \
	rm -f /root/cuda_11.0.3_450.51.06_linux.run


FROM BUILDBASE AS CUDA102
RUN touch /root/cuda102
COPY cuda_10.2.89_440.33.01_linux.run /root
RUN /bin/bash /root/cuda_10.2.89_440.33.01_linux.run --toolkit --silent --installpath=/usr/local/cuda-10.2 && \
	rm -f /root/cuda_10.2.89_440.33.01_linux.run

FROM BUILDBASE
RUN touch /root/cuda
COPY --from=CUDA110 /usr/local/cuda-11.0 /usr/local/cuda-11.0
COPY --from=CUDA102 /usr/local/cuda-10.2 /usr/local/cuda-10.2



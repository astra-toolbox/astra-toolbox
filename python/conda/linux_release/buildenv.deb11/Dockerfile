FROM debian:11 AS BUILDBASE
ENV DEBIAN_FRONTEND noninteractive
#RUN echo 'deb http://archive.debian.org/debian/ wheezy main' > /etc/apt/sources.list && echo 'deb http://archive.debian.org/debian-security/ wheezy/updates main' >> /etc/apt/sources.list && apt-get -o Acquire::Check-Valid-Until=false update && apt-get install -y perl-modules build-essential autoconf libtool automake libboost-dev git libxml2 && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y perl-modules build-essential autoconf libtool automake libboost-dev git libxml2 && rm -rf /var/lib/apt/lists/*

ENV PATH /root/miniconda3/bin:$PATH
COPY Miniconda3-py39_4.10.3-Linux-x86_64.sh /root/
RUN /bin/bash /root/Miniconda3-py39_4.10.3-Linux-x86_64.sh -b && \
	rm -f /root/Miniconda3*

RUN conda install -y conda-build conda-verify

FROM BUILDBASE AS CUDA117
RUN touch /root/cuda117
COPY cuda_11.7.0_515.43.04_linux.run /root
RUN /bin/bash /root/cuda_11.7.0_515.43.04_linux.run --toolkit --silent --installpath=/usr/local/cuda-11.7 && \
	rm -f /root/cuda_11.7.0_515.43.04_linux.run

FROM BUILDBASE AS CUDA116
RUN touch /root/cuda116
COPY cuda_11.6.0_510.39.01_linux.run /root
RUN /bin/bash /root/cuda_11.6.0_510.39.01_linux.run --toolkit --silent --installpath=/usr/local/cuda-11.6 && \
	rm -f /root/cuda_11.6.0_510.39.01_linux.run

FROM BUILDBASE AS CUDA115
RUN touch /root/cuda115
COPY cuda_11.5.1_495.29.05_linux.run /root
RUN /bin/bash /root/cuda_11.5.1_495.29.05_linux.run --toolkit --silent --installpath=/usr/local/cuda-11.5 && \
	rm -f /root/cuda_11.5.1_495.29.05_linux.run

FROM BUILDBASE AS CUDA114
RUN touch /root/cuda114
COPY cuda_11.4.1_470.57.02_linux.run /root
RUN /bin/bash /root/cuda_11.4.1_470.57.02_linux.run --toolkit --silent --installpath=/usr/local/cuda-11.4 && \
	rm -f /root/cuda_11.4.1_470.57.02_linux.run


FROM BUILDBASE AS CUDA113
RUN touch /root/cuda113
COPY cuda_11.3.1_465.19.01_linux.run /root
RUN /bin/bash /root/cuda_11.3.1_465.19.01_linux.run --toolkit --silent --installpath=/usr/local/cuda-11.3 && \
	rm -f /root/cuda_11.3.1_465.19.01_linux.run

FROM BUILDBASE AS CUDA112
RUN touch /root/cuda112
COPY cuda_11.2.2_460.32.03_linux.run /root
RUN /bin/bash /root/cuda_11.2.2_460.32.03_linux.run --toolkit --silent --installpath=/usr/local/cuda-11.2 && \
	rm -f /root/cuda_11.2.2_460.32.03_linux.run

FROM BUILDBASE
RUN touch /root/cuda
COPY --from=CUDA117 /usr/local/cuda-11.7 /usr/local/cuda-11.7
COPY --from=CUDA116 /usr/local/cuda-11.6 /usr/local/cuda-11.6
COPY --from=CUDA115 /usr/local/cuda-11.5 /usr/local/cuda-11.5
COPY --from=CUDA114 /usr/local/cuda-11.4 /usr/local/cuda-11.4
COPY --from=CUDA113 /usr/local/cuda-11.3 /usr/local/cuda-11.3
COPY --from=CUDA112 /usr/local/cuda-11.2 /usr/local/cuda-11.2

RUN conda create -y -n prep -c nvidia --download-only cudatoolkit=11.7 && \
    conda create -y -n prep -c nvidia --download-only cudatoolkit=11.6 && \
    conda create -y -n prep -c nvidia --download-only cudatoolkit=11.5 && \
    conda create -y -n prep -c nvidia --download-only cudatoolkit=11.4 && \
    conda create -y -n prep -c nvidia --download-only cudatoolkit=11.3 && \
    conda create -y -n prep -c nvidia --download-only "cudatoolkit=11.2.*,<11.2.72"

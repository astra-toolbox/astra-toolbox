FROM debian:7
ENV PATH /root/miniconda3/bin:$PATH
ENV DEBIAN_FRONTEND noninteractive
# http://developer.download.nvidia.com/compute/cuda/5_5/rel/installers/cuda_5.5.22_linux_64.run
ADD cuda_5.5.22_linux_64.run /root/
# https://repo.continuum.io/miniconda/Miniconda3-4.3.27.1-Linux-x86_64.sh
ADD Miniconda3-4.3.27.1-Linux-x86_64.sh /root/
RUN apt-get update
RUN apt-get install -y perl-modules build-essential autoconf libtool automake libboost-dev git
RUN /bin/bash /root/Miniconda3-4.3.27.1-Linux-x86_64.sh -b
RUN /bin/bash /root/cuda_5.5.22_linux_64.run -toolkit -silent
RUN conda install -y conda-build
ENV CUDA_ROOT /usr/local/cuda
ENV CC gcc
ENV CXX g++

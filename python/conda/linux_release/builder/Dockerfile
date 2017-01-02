FROM astra-build-env
ARG BUILD_NUMBER=
WORKDIR /root
RUN git clone --depth 1 https://github.com/astra-toolbox/astra-toolbox
RUN [ -z $BUILD_NUMBER ] || perl -pi -e "s/^(\s*number:\s*)[0-9]+$/\${1}$BUILD_NUMBER/" astra-toolbox/python/conda/libastra/meta.yaml astra-toolbox/python/conda/meta.yaml
RUN conda-build --python=3.5 astra-toolbox/python/conda/libastra
RUN conda-build --python 2.7 --numpy 1.8 astra-toolbox/python/conda
RUN conda-build --python 2.7 --numpy 1.9 astra-toolbox/python/conda
RUN conda-build --python 2.7 --numpy 1.10 astra-toolbox/python/conda
RUN conda-build --python 2.7 --numpy 1.11 astra-toolbox/python/conda
RUN conda-build --python 3.5 --numpy 1.9 astra-toolbox/python/conda
RUN conda-build --python 3.5 --numpy 1.10 astra-toolbox/python/conda
RUN conda-build --python 3.5 --numpy 1.11 astra-toolbox/python/conda
RUN conda-build --python 3.6 --numpy 1.11 astra-toolbox/python/conda

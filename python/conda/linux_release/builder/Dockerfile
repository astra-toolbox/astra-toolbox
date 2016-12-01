FROM astra-build-env
ARG BUILD_NUMBER=
WORKDIR /root
RUN git clone -b conda_release https://github.com/astra-toolbox/astra-toolbox
RUN [ -z $BUILD_NUMBER ] || perl -pi -e "s/^(\s*number:\s*)[0-9]+$/\${1}$BUILD_NUMBER/" astra-toolbox/python/conda/libastra/meta.yaml astra-toolbox/python/conda//meta.yaml
RUN conda-build --python=3.5 astra-toolbox/python/conda/libastra
RUN conda-build --python=3.5 astra-toolbox/python/conda
RUN conda-build --python=2.7 astra-toolbox/python/conda

#!/bin/sh

set -e

BRANCH=master
URL=https://github.com/astra-toolbox/astra-toolbox

echo "Cloning from ${URL}"
echo "        branch: ${BRANCH}"

cd /root
git clone --depth 1 --branch ${BRANCH} ${URL}

[ $# -eq 0 ] || perl -pi -e "s/^(\s*version:\s*)[0-9a-z+\.']+$/\${1}'$1'/" astra-toolbox/python/conda/libastra/meta.yaml astra-toolbox/python/conda/astra-toolbox/meta.yaml
[ $# -eq 0 ] || perl -pi -e "s/^(\s*number:\s*)[0-9]+$/\${1}$2/" astra-toolbox/python/conda/libastra/meta.yaml astra-toolbox/python/conda/astra-toolbox/meta.yaml
[ $# -eq 0 ] || perl -pi -e "s/^(\s*string:.+_)[0-9]+/\${1}$2/" astra-toolbox/python/conda/libastra/meta.yaml
[ $# -eq 0 ] || perl -pi -e "s/^(\s*-\s*libastra\s*==\s*)[0-9a-z+\.]+$/\${1}$1/" astra-toolbox/python/conda/astra-toolbox/meta.yaml


conda-build -m astra-toolbox/python/conda/libastra/linux_build_config.yaml astra-toolbox/python/conda/libastra

conda-build -m astra-toolbox/python/conda/astra-toolbox/linux_build_config.yaml astra-toolbox/python/conda/astra-toolbox

cp /root/miniconda3/conda-bld/linux-64/*astra* /out

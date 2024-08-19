#!/bin/sh

set -e

BRANCH=master
URL=https://github.com/astra-toolbox/astra-toolbox

echo "Cloning from ${URL}"
echo "        branch: ${BRANCH}"

cd /root
git clone --depth 1 --branch ${BRANCH} ${URL}

echo "Setting conda git_url/git_tag to match:"

perl -pi -e "s~^(\s*git_url:\s*)https://github.com/.+$~\${1}${URL}~" astra-toolbox/build/conda/libastra/meta.yaml astra-toolbox/build/conda/astra-toolbox/meta.yaml
perl -pi -e "s/^(\s*git_tag:\s*).+$/\${1}${BRANCH}/" astra-toolbox/build/conda/libastra/meta.yaml astra-toolbox/build/conda/astra-toolbox/meta.yaml

grep git_ astra-toolbox/build/conda/libastra/meta.yaml

[ $# -eq 0 ] || echo "Setting version-build to $1-$2"

[ $# -eq 0 ] || perl -pi -e "s/^(\s*version:\s*)[0-9a-z+\.']+$/\${1}'$1'/" astra-toolbox/build/conda/libastra/meta.yaml astra-toolbox/build/conda/astra-toolbox/meta.yaml
[ $# -eq 0 ] || perl -pi -e "s/^(\s*number:\s*)[0-9]+$/\${1}$2/" astra-toolbox/build/conda/libastra/meta.yaml astra-toolbox/build/conda/astra-toolbox/meta.yaml
[ $# -eq 0 ] || perl -pi -e "s/^(\s*-\s*libastra\s*==\s*)[0-9a-z+\.]+$/\${1}$1\${2}/" astra-toolbox/build/conda/astra-toolbox/meta.yaml
[ $# -eq 0 ] || perl -pi -e "s/^(\s*-\s*libastra\s*==\s*)[0-9a-z+\.]+(\s+[0-9a-z_+\.\*]+\s+#.*)$/\${1}$1\${2}/" astra-toolbox/build/conda/astra-toolbox/meta.yaml

grep 'version:\|number:' astra-toolbox/build/conda/libastra/meta.yaml

CONF=linux_$3_build_config.yaml

conda build --no-test -c nvidia -m astra-toolbox/build/conda/libastra/${CONF} astra-toolbox/build/conda/libastra

[ x$4 = xfull ] && conda build -c nvidia -m astra-toolbox/build/conda/astra-toolbox/${CONF} astra-toolbox/build/conda/astra-toolbox

[ x$4 = xfull ] && conda build --test -c nvidia /root/miniconda3/conda-bld/linux-64/libastra*tar.bz2

cp /root/miniconda3/conda-bld/linux-64/*astra* /out

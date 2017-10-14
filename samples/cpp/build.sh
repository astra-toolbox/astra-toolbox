#!/bin/bash

in_array() {
    local hay needle=$1 i=0
    shift
    for hay; do
        if $(echo "$hay" | grep -q "$needle"); then
	    echo $i
	    return 0
	else
	    ((i++))
	fi
    done
    echo -1
    return 1
}

all_names=(s001_sinogram_par2d
	   s003_gpu_reconstruction
	   s004_cpu_reconstruction
	   s010_supersampling
	   s012_masks
	   s013_constraints
	   s014_FBP
	   s019_experimental_multires)

if [ -z $1 ]; then
    names=(${all_names[@]})
    options=(${all_opts[@]})
else
    if [ "$1" == "clean" ]; then
	rm -f ${all_names[@]}
    else
	idx=$(in_array $1 ${all_names[@]})
	[ $idx -ge 0 ] && names=(${all_names[$idx]}) || (echo "Target not found: $1" ; exit)
    fi
fi

shift

for name in ${names[@]}; do
    echo -ne $name
    g++ -o $name $name.cpp \
	../../cpp/creators.o ../../lib/libastra.a \
	-I../../ -I../../include -I/opt/cuda/include \
	-L/opt/cuda/lib64 -Wall \
	-lcudart -lcufft -pthread \
	-march=native -O3 -lz \
	-DASTRA_CUDA $@
    strip $name
    echo " [DONE]"
done

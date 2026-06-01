import numpy as np
import os
import msgpack
import msgpack_numpy
import sys
import argparse

msgpack_numpy.patch()

MIN_SUPPORTED_FORMAT_VERSION = 3
MAX_SUPPORTED_FORMAT_VERSION = 3

from plot import show

def read_test_file(filename):
    with open(filename, 'rb') as F:
        data = msgpack.load(F)
    if 'version' not in data or data['version'] < MIN_SUPPORTED_FORMAT_VERSION or data['version'] > MAX_SUPPORTED_FORMAT_VERSION:
        raise ValueError("Unsupported file: " + filename)
    return data

def inspect_FP(name, data, reference):
    print("projector config:", data['fp_proj_cfg'])
    if reference is None:
        show(name, (data['fp_in'],), ("input",))
    else:
        show(name, (data['fp_in'], reference['data']), ("input", "output"))

def inspect_BP(name, data, reference):
    print("projector config:", data['bp_proj_cfg'])
    if reference is None:
        show(name, (data['bp_in'],), ("input",))
    else:
        show(name, (data['bp_in'], reference['data']), ("input", "output"))

def inspect_rec(name, data, reference):
    # TODO: Allow linking the slicers for masks with input/output
    # (Instead of linking either everything or nothing)
    print("projector config:", data['rec_proj_cfg'])
    print(f"x0: {data['rec_x0']}")
    print(f"iters: {data['rec_iters']}")
    vols = [ data['rec_in'] ]
    titles = [ "input" ]
    if 'rec_vol_mask' in data:
        vols.append(data['rec_vol_mask'])
        titles.append("volume mask")
    if 'rec_proj_mask' in data:
        vols.append(data['rec_proj_mask'])
        titles.append("projection mask")
    if reference is not None:
        vols.append(reference['data'])
        titles.append("output")
    show(name, vols, titles)


runners = { 'fp2d': inspect_FP, 'fp3d': inspect_FP,
            'bp2d': inspect_BP, 'bp3d': inspect_BP,
            'rec2d': inspect_rec, 'rec3d': inspect_rec }

def inspect_test(filename, reference_dir):
    data = read_test_file(filename)
    ret = True

    if data['type'][:3] != 'in_':
        raise ValueError("Invalid input type")
    test_type = data['type'][3:]
    name = os.path.splitext(os.path.basename(filename))[0]

    print(f"Test {test_type} name {name}")

    reference = None
    if reference_dir is not None:
        reference_filename = os.path.join('outputs', reference_dir, test_type + '_' + name + '.mp')
        try:
            reference = read_test_file(reference_filename)
            print(f"Showing output from {reference_filename}")
        except:
            pass

    runner = runners[test_type]
    runner(test_type + '_' + name, data, reference)

if __name__ == '__main__':
     parser = argparse.ArgumentParser(
             prog='inspect_data.py',
             description='ASTRA regression test inspector')
     parser.add_argument('--reference', help="name of output to inspect (optional)")
     parser.add_argument('filename', help="test input file")
     args = parser.parse_args()

     inspect_test(args.filename, args.reference)


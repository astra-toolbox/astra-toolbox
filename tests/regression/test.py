import numpy as np
import astra
import time
import os
import msgpack
import msgpack_numpy
import argparse
import sys
import traceback
import multiprocessing

msgpack_numpy.patch()

FORMAT_VERSION = 3
MIN_SUPPORTED_FORMAT_VERSION = 3
MAX_SUPPORTED_FORMAT_VERSION = 3


#ATOL = 1e-8
#RTOL = 1e-5
ATOL = 1e-1
RTOL = 1e-1
REPEAT = 5
PLOT = False
QUIET = False
BENCH = False

def read_test_file(basename):
    with open(basename + '.mp', 'rb') as F:
        data = msgpack.load(F)
    if 'version' not in data or data['version'] < MIN_SUPPORTED_FORMAT_VERSION or data['version'] > MAX_SUPPORTED_FORMAT_VERSION:
        raise ValueError("Unsupported file: " + basename + '.mp')
    return data

def compare_results(data, ref):
    try:
        return np.allclose(data, ref, rtol=RTOL, atol=ATOL, equal_nan=True)
    except:
        return False

def run_FP2D(data, repeat=1):
    projector_id = astra.projector.create(data['fp_proj_cfg'])
    data_in = data['fp_in'].copy()
    for r in range(repeat):
        # There are nicer ways to run this in current astra, but we want to
        # support older versions to compare against.
        output_id, output = astra.create_sino(data_in, projector_id, returnData=True)
        astra.data2d.delete(output_id)
    astra.projector.delete(projector_id)
    return output

def run_FP3D(data, repeat=1):
    projector_id = astra.projector3d.create(data['fp_proj_cfg'])
    data_in = data['fp_in'].copy()
    for r in range(repeat):
        # There are nicer ways to run this in current astra, but we want to
        # support older versions to compare against.
        output_id, output = astra.create_sino3d_gpu(data_in, data['fp_proj_cfg']['ProjectionGeometry'], data['fp_proj_cfg']['VolumeGeometry'], returnData=True)
        astra.data3d.delete(output_id)
    astra.projector3d.delete(projector_id)
    return output

def run_BP2D(data, repeat=1):
    projector_id = astra.projector.create(data['bp_proj_cfg'])
    data_in = data['bp_in'].copy()
    for r in range(repeat):
        # There are nicer ways to run this in current astra, but we want to
        # support older versions to compare against.
        output_id, output = astra.create_backprojection(data_in, projector_id, returnData=True)
        astra.data2d.delete(output_id)
    astra.projector.delete(projector_id)
    return output

def run_BP3D(data, repeat=1):
    #print(data['bp_proj_cfg']['VolumeGeometry'])
    #print(data['bp_proj_cfg']['ProjectionGeometry'])
    projector_id = astra.projector3d.create(data['bp_proj_cfg'])
    data_in = data['bp_in'].copy()
    for r in range(repeat):
        # There are nicer ways to run this in current astra, but we want to
        # support older versions to compare against.
        output_id, output = astra.create_backprojection3d_gpu(data['bp_in'].copy(), data['bp_proj_cfg']['ProjectionGeometry'], data['bp_proj_cfg']['VolumeGeometry'], returnData=True)
        astra.data3d.delete(output_id)
    astra.projector3d.delete(projector_id)
    return output

def run_rec2D(data, repeat=1):
    projector_id = astra.projector.create(data['rec_proj_cfg'])
    sino_id = astra.data2d.create('-sino', data['rec_proj_cfg']['ProjectionGeometry'], data['rec_in'].copy())
    rec_id = astra.data2d.create('-vol', data['rec_proj_cfg']['VolumeGeometry'], data['rec_x0'])
    if 'rec_vol_mask' in data:
        vol_mask_id = astra.data2d.create('-vol', data['rec_proj_cfg']['VolumeGeometry'], data['rec_vol_mask'].copy())
    if 'rec_proj_mask' in data:
        proj_mask_id = astra.data2d.create('-sino', data['rec_proj_cfg']['ProjectionGeometry'], data['rec_proj_mask'].copy())


    cfg = data['rec_cfg']
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectorId'] = projector_id
    if 'options' not in cfg:
        cfg['options'] = {}
    if 'rec_vol_mask' in data:
        cfg['options']['ReconstructionMaskId'] = vol_mask_id
    if 'rec_proj_mask' in data:
        cfg['options']['SinogramMaskId'] = proj_mask_id
    alg_id = astra.algorithm.create(cfg)
    iters = data['rec_iters']

    for r in range(repeat):
        if isinstance(iters, tuple) or isinstance(iters, list):
            for i in iters:
                astra.algorithm.run(alg_id, i)
        else:
            astra.algorithm.run(alg_id, iters)

        if r == 0:
            output = astra.data2d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data2d.delete([rec_id, sino_id])
    astra.projector.delete(projector_id)
    if 'rec_vol_mask' in data:
        astra.data2d.delete(vol_mask_id)
    if 'rec_proj_mask' in data:
        astra.data2d.delete(proj_mask_id)

    return output


def run_rec3D(data, repeat=1):
    projector_id = astra.projector3d.create(data['rec_proj_cfg'])
    sino_id = astra.data3d.create('-proj3d', data['rec_proj_cfg']['ProjectionGeometry'], data['rec_in'].copy())
    rec_id = astra.data3d.create('-vol', data['rec_proj_cfg']['VolumeGeometry'], data['rec_x0'])
    if 'rec_vol_mask' in data:
        vol_mask_id = astra.data3d.create('-vol', data['rec_proj_cfg']['VolumeGeometry'], data['rec_vol_mask'].copy())
    if 'rec_proj_mask' in data:
        proj_mask_id = astra.data3d.create('-proj3d', data['rec_proj_cfg']['ProjectionGeometry'], data['rec_proj_mask'].copy())

    cfg = data['rec_cfg']
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectorId'] = projector_id
    if 'options' not in cfg:
        cfg['options'] = {}
    if 'rec_vol_mask' in data:
        cfg['options']['ReconstructionMaskId'] = vol_mask_id
    if 'rec_proj_mask' in data:
        cfg['options']['SinogramMaskId'] = proj_mask_id
    alg_id = astra.algorithm.create(cfg)
    iters = data['rec_iters']

    for r in range(repeat):
        if isinstance(iters, tuple) or isinstance(iters, list):
            for i in iters:
                astra.algorithm.run(alg_id, i)
        else:
            astra.algorithm.run(alg_id, iters)

        if r == 0:
            output = astra.data3d.get(rec_id)

    astra.algorithm.delete(alg_id)
    astra.data3d.delete([rec_id, sino_id])
    astra.projector.delete(projector_id)

    return output


runners = { 'fp2d': run_FP2D, 'fp3d': run_FP3D,
            'bp2d': run_BP2D, 'bp3d': run_BP3D,
            'rec2d': run_rec2D, 'rec3d': run_rec3D }
#runners = { 'fp3d': run_FP3D,
#            'bp3d': run_BP3D,
#            'rec3d': run_rec3D }
#runners = { 'bp3d': run_BP3D }

def run_test(test_type, name, output_dir, reference_dir, quiet, plot, bench):
    filename = os.path.join('inputs', test_type, name)
    data = read_test_file(filename)
    ret = True

    if reference_dir is not None:
        reference_filename = os.path.join('outputs', reference_dir, test_type + '_' + name)
        try:
            reference = read_test_file(reference_filename)
        except:
            reference_dir = None

    runner = runners[test_type]
    try:
        # Get test result
        # This also acts as a single warm-up run when benchmarking
        result = runner(data, 1)
        if bench:
            time_start = time.perf_counter()
            runner(data, REPEAT)
            time_end = time.perf_counter()
    except Exception as e:
        if not quiet:
            print(traceback.format_exc())
            print('Fail with exception ' + test_type + '_' + name)
        sys.exit(1)

    if bench:
        our_time = (time_end - time_start) / REPEAT

    if reference_dir is not None:
        if not compare_results(result, reference['data']):
            if not quiet:
                print('Fail ' + test_type + '_' + name)
            if plot:
                from plot import compare
                #compare(test_type + '_' + name, (result, reference['data'], 1.0 + ((result - reference['data']) / reference['data'])), ("result", "reference", "(result - reference)/reference"))
                compare(test_type + '_' + name, (result, reference['data'], (result - reference['data'])), ("result", "reference", "(result - reference)"))
            ret = False
        else:
            if not quiet:
                print('Pass ' + test_type + '_' + name)
        if bench:
            # TODO: Maybe check reference['gpu_info'] == astra.get_gpu_info() ?
            if 'time' in reference:
                ref_time = reference['time']
                diff_time = 100 * (our_time - ref_time) / ref_time
                sign = "+" if diff_time > 0 else ""
                print(f"Bench {1000*ref_time:4.0f}ms -> {1000*our_time:4.0f}ms    {sign}{diff_time:.2f}%")
            else:
                print(f"Bench {1000*our_time:4.0f}ms (no reference time available)")
    else:
        if not quiet:
            print('Run ' + test_type + '_' + name)
            if bench:
                print(f"Bench {1000*our_time:4.0f}ms")

    if output_dir is not None:
        output_filename = os.path.join('outputs', output_dir, test_type + '_' + name + '.mp')
        data = { 'data': result, 'version': FORMAT_VERSION }
        if bench:
            data['time'] = our_time
        data['type'] = 'out_' + test_type
        try:
            data['astra_version'] = astra.__version__
        except:
            pass
        try:
            data['gpu_info'] = astra.get_gpu_info()
        except:
            pass
        with open(output_filename, 'wb') as F:
            msgpack.pack(data, F)

    if not ret:
        sys.exit(1)

def run_tests_of_type(test_type, output_dir, reference_dir):
    if output_dir is not None:
        try:
            os.mkdir(os.path.join('outputs', output_dir))
        except FileExistsError:
            pass
    d = os.path.join('inputs', test_type)
    #tests = [x[:-4] for x in os.listdir(d) if x.endswith('.mat') ] + [x[:-3] for x in os.listdir(d) if x.endswith('.mp') ]
    tests = [x[:-3] for x in os.listdir(d) if x.endswith('.mp') ]

    count = 0
    failed = 0
    for name in tests:
        p = multiprocessing.Process(target=run_test, args=(test_type, name, output_dir, reference_dir, QUIET, PLOT, BENCH))
        p.start()
        p.join()
        if p.exitcode < 0:
            print(f'Crash {test_type}_{name} (signal {-p.exitcode})' )

        count += 1
        if p.exitcode != 0:
            failed += 1

    return (count, failed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            prog='test.py',
            description='ASTRA regression tests')
    parser.add_argument('--output', help="name of output for this run (optional)")
    parser.add_argument('--reference', help="name of reference output (optional)")
    parser.add_argument('--silent', help="do not output individual test results (optional)", action='store_true')
    parser.add_argument('--display', help="display outputs of failing tests (optional)", action='store_true')
    parser.add_argument('--benchmark', help="benchmark the tests against reference (optional)", action='store_true')
    args = parser.parse_args()

    if args.silent:
        QUIET = True
    if args.display:
        PLOT = True
    if args.benchmark:
        BENCH = True

    count = 0
    failed = 0
    for test_type in runners.keys():
        res = run_tests_of_type(test_type, args.output, args.reference)
        count += res[0]
        failed += res[1]

    print(f"Failed {failed} out of {count} tests")

    if failed > 0:
        sys.exit(1)

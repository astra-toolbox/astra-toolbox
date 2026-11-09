Dependencies:

astra-toolbox
msgpack
msgpack-numpy

Extra for plotting:

pyqtgraph
pyqt6



Usage:

Run generate_data2d.py and generate_data3d.py to create test inputs.

Activate conda environment with astra 2.5.0 (say), and run

$ python test.py --output astra-2.5.0

This runs all tests, and reference benchmarks, and stores outputs in the
directory outputs/astra-2.5.0 .

Activate conda environment astra master (say), and run

$ python test.py --reference astra-2.5.0

This runs all tests, and compares outputs with the previously stored outputs.
You can also combine --output and --reference to store the new outputs.

There is an option --display to visually compare any failing test outputs.
The comparison window has left/right shortcuts to scroll through slices, and
'q' to close the window.

There is an option --bench to additionally run basic benchmarks on these tests.
Use it with both the --output and the --reference run to compare benchmarks.


The utility inspect_data.py can be used to inspect inputs and outputs.
Example usage

$ python inspect_data.py inputs/bp2d/fan_cuda_128.mp
$ python inspect_data.py inputs/bp2d/fan_cuda_128.mp --reference astra-2.5.0



Assorted notes:

These tests are not currently intended to be run unsupervised in CI pipelines.

Many tests expect a GPU.

Keep tests to at most a few seconds, with a total runtime of at most a few
minutes, to keep it efficient to run all tests regularly.

These regression tests are intended to be used in tandem with our other test
suite that also compares outputs against known-good results.

There is a benchmarking feature, but this is not intended to benchmark scaling
with volume size, or of large volumes in general. It is only intended to catch
unexpected differences in speed.

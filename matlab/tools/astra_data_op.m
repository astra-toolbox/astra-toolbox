function astra_data_op(op, data, scalar, gpu_core)

cfg = astra_struct('DataOperation_CUDA');
cfg.Operation = op;
cfg.Scalar = scalar;
cfg.DataId = data;
cfg.option.GPUindex = gpu_core;

alg_id = astra_mex_algorithm('create',cfg);
astra_mex_algorithm('run',alg_id);
astra_mex_algorithm('delete',alg_id);
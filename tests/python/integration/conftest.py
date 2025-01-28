import pytest
import astra


cuda_present = astra.use_cuda()

cupy_present = False
if cuda_present:
    try:
        import cupy
        cupy_present = True
    except Exception:
        pass

pytorch_present = False
try:
    import torch
    pytorch_present = True
except Exception:
    pass

pytorch_cuda_present = False
if pytorch_present:
    try:
        import torch
        if torch.cuda.is_available():
            pytorch_cuda_present = True
    except Exception:
        pass

jax_present = False
try:
    import jax
    jax_present = True
except Exception:
    pass

jax_cuda_present = False
if jax_present:
    try:
        import jax
        if len(jax.devices('cuda')) > 0:
            jax_cuda_present = True
    except Exception:
        pass

backends_to_skip = []

if not cupy_present:
    backends_to_skip.append('cupy')

if not jax_present:
    backends_to_skip.append('jax_cpu')
if not jax_cuda_present:
    backends_to_skip.append('jax_cuda')

if not pytorch_present:
    backends_to_skip.append('pytorch_cpu')
if not pytorch_cuda_present:
    backends_to_skip.append('pytorch_cuda')

def pytest_collection_modifyitems(config, items):
    for item in items:
        if hasattr(item, 'callspec') and item.callspec.params.get('backend') in backends_to_skip:
            item.add_marker(pytest.mark.skip('Backend skipped because it is unavailable'))

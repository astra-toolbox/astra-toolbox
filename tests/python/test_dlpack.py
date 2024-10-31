import astra
import torch
import numpy as np

vol = np.random.randn(4,5,6).astype(np.float32)
vg = astra.create_vol_geom(5,6,4)

vid = astra.data3d.link('-vol', vg, vol)
print(vid)
vtest = astra.data3d.get(vid)
print(vtest - vol)


vol = torch.tensor(vol, device='cuda')
vid2 = astra.data3d.link('-vol', vg, vol)
print(vid2)


pg = astra.create_proj_geom('parallel3d', 1, 1, 5, 5, [0, np.pi/4])

_, data1 = astra.create_sino3d_gpu(vid, pg, vg, returnData=True)
_, data2 = astra.create_sino3d_gpu(vid2, pg, vg, returnData=True)

print(data1 - data2)

# errors

#vol = np.random.randn(4,5,7).astype(np.float32)
#vg = astra.create_vol_geom(5,6,4)
#vid = astra.data3d.link('-vol', vg, vol)

#vol = np.random.randn(4,5,6).astype(np.float64)
#vg = astra.create_vol_geom(5,6,4)
#vid = astra.data3d.link('-vol', vg, vol)

#vol = np.transpose(np.random.randn(6,5,4).astype(np.float32),(2,1,0))
#vg = astra.create_vol_geom(5,6,4)
#vid = astra.data3d.link('-vol', vg, vol)

#vg = astra.create_vol_geom(5,6,4)
#vid = astra.data3d.link('-vol', vg, 5)

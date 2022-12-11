import numpy as np
import open3d as o3d
import tqdm
import sys
import time

################################################################################
# Example usage: python util_get_grids.py longdress_vox10_1300.ply 5 ###########
################################################################################
qstr = ''
fid = sys.argv[1].split('/')[-1][:-4]
if len(sys.argv) == 3:
    lx = int(sys.argv[2])
else:
    lx = 5
origins = np.loadtxt(f'{fid}_l{lx}{qstr}_origins.txt', delimiter=',')
np.save(f'{fid}_l{lx}{qstr}_origins', origins)

cube_template = np.zeros((32,32,32,3), dtype=np.int64)
for i in range(32):
    cube_template[i,:,:,0] = i
    for j in range(32):
        cube_template[i,j,:,1] = j
        cube_template[i,j,:,2] = np.linspace(0, 31, 32)

pts = []
for i, origin in enumerate(origins):
    cube = cube_template + origin
    pts.append(cube)
pts = np.array(pts).reshape(-1, 3)

pcd = o3d.io.read_point_cloud(sys.argv[1])
kd_tree = o3d.geometry.KDTreeFlann(pcd)

start = time.time()
res = np.zeros((len(pts),), dtype=int)
for i,p in enumerate(pts):
    res[i] = kd_tree.search_knn_vector_3d(p, 1)[1][0]
end = time.time()
refs = np.asarray(pcd.points)
dist = np.sqrt(np.sum(np.square(refs[res] - pts), -1))
dist = dist.reshape(len(origins),1,32,32,32)
gt_grid = (dist == 0).astype(np.uint8)
gt_grid = gt_grid.reshape(dist.shape)
np.save(f'{fid}_l{lx}{qstr}_gt_grid', gt_grid)
np.save(f'{fid}_l{lx}{qstr}_dist', dist)

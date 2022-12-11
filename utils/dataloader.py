import numpy as np
import open3d as o3d
import torch

def loadCloudFromBinary(file, cols=3):
    f = open(file, "rb")
    binary_data = f.read()
    f.close()
    temp = np.frombuffer(
        binary_data, dtype='float32', count=-1)
    data = np.reshape(temp, (cols, int(temp.size/cols)))
    return data.T[:,:3]

def loadCloudFromPly(file, return_pcd=False):
    pcd = o3d.io.read_point_cloud(file)
    d = np.asarray(pcd.points)
    if return_pcd:
        return pcd,d
    else:
        return d


class ContinuousVoxelDataset(torch.utils.data.Dataset):
    def __init__(self, pts_fn, bit_depth, max_depth):
        super(ContinuousVoxelDataset, self).__init__()
        if 'bin' in pts_fn:
            npd = loadCloudFromBinary(pts_fn, 11)
        elif 'ply' in pts_fn:
            pcd, npd = loadCloudFromPly(pts_fn, True)
            self.pcd = pcd
        # Normalize NPD
        # means = np.mean(npd, 0)
        # self.means = means
        # self.npd = npd - means

        
        anchor_pts = np.array([
            [0,0,0],[0,0,1],[0,1,0],[0,1,1],
            [1,0,0],[1,0,1],[1,1,0],[1,1,1],
        ], dtype=npd.dtype)

        anchor_pts = (anchor_pts *1023)
        npd = np.concatenate([npd, anchor_pts], 0)
        self.npd = npd

        print('Initialize dataset')

        self.N = self.npd.shape[0]
        print('Number of points ', self.N)
        # self.pcd = o3d.geometry.PointCloud()

        self.pcd.points = o3d.utility.Vector3dVector(self.npd)
        # self.scale = np.max(self.pcd.get_max_bound() - self.pcd.get_min_bound())
        # self.center = self.pcd.get_center()
        # self.pcd.scale(1 / np.max(self.pcd.get_max_bound() - self.pcd.get_min_bound()), center=self.pcd.get_center())

        # Build Octree
        print('Building Octree')
        self.octree = o3d.geometry.Octree(max_depth=5)
        print('Building Octree')
        self.octree.convert_from_point_cloud(self.pcd, size_expand=0)
        print('Octree built')

        self.leaf_nodes = []
        self.leaf_node_info = []
        def f_traverse(node, node_info):
            if isinstance(node, o3d.geometry.OctreeLeafNode):
                self.leaf_nodes.append(node)
                self.leaf_node_info.append(node_info)
            return False
        self.octree.traverse(f_traverse)

        print('Number of leaf nodes: %d' % (len(self.leaf_nodes)))
        voxel_dim = bit_depth - max_depth
        nleaves = len(self.leaf_nodes)

        self.coord_list = []
        self.feat_list = []
        self.vox_grid_list = []
        ignored = 0

        remove_0 = True
        remove_1023 = True
        if len(np.where(npd[:-8] == 0)[0]) > 0:
            remove_0 = False
        if len(np.where(npd[:-8] == 1023)[0]) > 0:
            remove_1023 = False

        for i in range(nleaves):
            vox = o3d.geometry.PointCloud()
            vpts = np.asarray(pcd.points)[self.leaf_nodes[i].indices]
            if (np.min(vpts) == 0 and remove_0) or (np.max(vpts) == 1023 and remove_1023):
                ignored += 1
                print('Ignoring points ', vpts)
                continue
            # vcolors = np.asarray(pcd.colors)[self.leaf_nodes[i].indices]
            vox.points = o3d.utility.Vector3dVector(vpts)
            # vox.colors = o3d.utility.Vector3dVector(vcolors)
            s = self.leaf_node_info[i].size
            vs = s / (2**voxel_dim)
            vox_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(vox, vs)
            l = vox_grid.get_voxels()
            idxs = [v.grid_index for v in l]
            if len(idxs) == 1 and (np.min(vpts) == 0 or np.max(vpts) == 1023):
                ignored += 1
                continue
            self.vox_grid_list.append(vox_grid)
            nidxs = len(idxs)
            np_coords = np.array(idxs, dtype=np.float32)
            coords = torch.tensor(np_coords)
            feat = torch.ones((nidxs, 1)).float()
            self.coord_list.append(coords)
            self.feat_list.append(feat)
        
        self.N_leaf = nleaves - ignored

        # Additionally build a KD Tree
        cube_template = np.zeros((32,32,32,3), dtype=np.int64)
        for i in range(32):
            cube_template[i,:,:,0] = i
            for j in range(32):
                cube_template[i,j,:,1] = j
                cube_template[i,j,:,2] = np.linspace(0, 31, 32)
        pts = []
        for i in range(self.N_leaf):
            origin = np.array(self.vox_grid_list[i].origin).astype(int)
            cube = cube_template + origin
            pts.append(cube)
        pts = np.array(pts).reshape(-1, 3)
        self.kd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        res = np.zeros((len(pts),), dtype=int)
        for i,p in tqdm.tqdm(enumerate(pts)):
            res[i] = self.kd_tree.search_knn_vector_3d(p, 1)[1][0]
        refs = np.asarray(pcd.points)
        dist = np.sqrt(np.sum(np.square(refs[res] - pts), -1)).reshape(self.N_leaf, 1, 32, 32, 32)
        np.save(f'{pts_fn[:-4]}_dist', dist)
        import IPython
        IPython.embed()
        quit()
        dist = np.load(f'{pts_fn[:-4]}_dist.npy')
        self.dist = torch.from_numpy(dist).float()

    def __getitem__(self, idx):
        # Random sample 
        nidx = np.array([idx], dtype=np.float32)
        tidx = torch.tensor(nidx).long()
        return tidx, self.coord_list[idx], self.feat_list[idx], self.dist[idx]

    def __len__(self):
        return self.N_leaf

class LoadedVoxelDataset(torch.utils.data.Dataset):
    def __init__(self, origin_fn, gt_fn, dist_fn, shuffle=True):
        super(LoadedVoxelDataset, self).__init__()
        self.origins = np.load(origin_fn)
        self.gt_grid = np.load(gt_fn)
        self.dist = np.load(dist_fn)
        self.N_leaf = self.origins.shape[0]
        self.N = self.gt_grid.sum()
        print(f"A total numbero of {self.N} points")
        self.shuffle = shuffle

    def __getitem__(self, idx):
        # Random sample 
        if self.shuffle:
            magic_num = 2113
            idx = (idx * magic_num) % self.N_leaf
        nidx = np.array([idx], dtype=np.float32)
        tidx = torch.tensor(nidx).long()
        gt_grid_t = torch.from_numpy(self.gt_grid[idx]).float()
        dist_t = torch.from_numpy(self.dist[idx]).float()
        return tidx, gt_grid_t, dist_t
    
    def get_all(self):
        gt_grid_t = torch.from_numpy(self.gt_grid).float()
        dist_t = torch.from_numpy(self.dist).float()
        return gt_grid_t, dist_t


    def __len__(self):
        return self.N_leaf

class LoadedBaseVoxelDataset(torch.utils.data.Dataset):
    def __init__(self, origin_fn, gt_fn, dist_fn, base_fn, shuffle=True):
        super(LoadedBaseVoxelDataset, self).__init__()
        self.origins = np.load(origin_fn)
        self.gt_grid = np.load(gt_fn)
        self.dist = np.load(dist_fn)
        base = np.load(base_fn)
        self.base = np.pad(base, 2, mode='constant', constant_values=0)
        self.N_leaf = self.origins.shape[0]
        self.N = self.gt_grid.sum()
        print(f"A total number of {self.N} points")
        self.shuffle = shuffle
        bases = []
        for idx in range(self.N_leaf):
            org = self.origins[idx].astype(int)//32
            ctx = self.base[org[0]:org[0]+5,org[1]:org[1]+5,org[2]:org[2]+5].reshape(1,5,5,5)
            bases.append(ctx)
        self.bases = np.array(bases)

    def __getitem__(self, idx):
        # Random sample 
        if self.shuffle:
            magic_num = 2113
            idx = (idx * magic_num) % self.N_leaf
        nidx = np.array([idx], dtype=np.float32)
        tidx = torch.tensor(nidx).long()
        gt_grid_t = torch.from_numpy(self.gt_grid[idx]).float()
        dist_t = torch.from_numpy(self.dist[idx]).float()
        ctx_t = torch.from_numpy(self.bases[idx]).float()
        return tidx, gt_grid_t, dist_t, ctx_t
    
    def get_all(self):
        gt_grid_t = torch.from_numpy(self.gt_grid).float()
        dist_t = torch.from_numpy(self.dist).float()
        ctx_t = torch.from_numpy(self.bases).float()
        return gt_grid_t, dist_t, ctx_t

    def __len__(self):
        return self.N_leaf


class LoadedInSituBaseVoxelDataset(torch.utils.data.Dataset):
    def __init__(self, origin_fn, gt_fn, dist_fn, base_fn, shuffle=True):
        super(LoadedInSituBaseVoxelDataset, self).__init__()
        self.origins = np.load(origin_fn)
        self.gt_grid = np.load(gt_fn)
        self.dist = np.load(dist_fn)
        base = np.load(base_fn)
        self.base = np.pad(base, 2, mode='constant', constant_values=0)
        self.N_leaf = self.origins.shape[0]
        self.N = self.gt_grid.sum()
        print(f"A total number of {self.N} points")
        self.shuffle = shuffle
        bases = []
        for idx in range(self.N_leaf):
            org = self.origins[idx].astype(int)//32
            ctx = self.base[org[0]:org[0]+5,org[1]:org[1]+5,org[2]:org[2]+5].reshape(1,5,5,5)
            bases.append(ctx)
        self.bases = np.array(bases)

    def __getitem__(self, idx):
        # Random sample 
        if self.shuffle:
            magic_num = 2113
            idx = (idx * magic_num) % self.N_leaf
        nidx = np.array([idx], dtype=np.float32)
        tidx = torch.tensor(nidx).long()
        return tidx
    
    def get_all(self):
        gt_grid_t = torch.from_numpy(self.gt_grid).float()
        dist_t = torch.from_numpy(self.dist).float()
        ctx_t = torch.from_numpy(self.bases).float()
        return gt_grid_t, dist_t, ctx_t

    def __len__(self):
        return self.N_leaf
        
class LoadedVideoBaseVoxelDataset(torch.utils.data.Dataset):
    def __init__(self, origin_fs, gt_fs, dist_fs, base_fs):
        super(LoadedVideoBaseVoxelDataset, self).__init__()
        self.origins_list = []
        self.gt_grid_list = []
        self.dist_list = []
        self.base_list = []
        self.nleaf_list = []
        MAX_NLEAF = 2048
        self.inverse_LUT = np.zeros((len(origin_fs)*MAX_NLEAF), dtype=int)
        ptr = 0
        cnt = 0
        for origin_fn, gt_fn, dist_fn, base_fn in zip(origin_fs, gt_fs, dist_fs, base_fs):
            origin = np.load(origin_fn)
            gt = np.load(gt_fn)
            dist = np.load(dist_fn)
            base = np.load(base_fn)
            base = np.pad(base, 2, mode='constant', constant_values=0)
            assert origin.shape[0] == gt.shape[0]
            assert gt.shape[0] == dist.shape[0]
            self.origins_list.append(origin)
            self.gt_grid_list.append(gt)
            self.dist_list.append(dist)
            self.base_list.append(base)
            N = origin.shape[0]
            self.inverse_LUT[ptr:ptr+N] = cnt
            self.nleaf_list.append(N)
            ptr = ptr + N
            cnt += 1

        self.origins = np.concatenate(self.origins_list, 0)
        self.gt_grid = np.concatenate(self.gt_grid_list, 0)
        self.dist = np.concatenate(self.dist_list, 0)

        self.bases = np.stack(self.base_list, 0).astype(int)

        self.N_obj = cnt

        self.N_leaf = self.origins.shape[0]
        self.N = self.gt_grid.sum()
        print(f"A total numbero of {self.N} points")

    def __getitem__(self, idx):
        nidx = np.array([idx], dtype=np.float32)
        tidx = torch.tensor(nidx).long()
        gt_grid_t = torch.from_numpy(self.gt_grid[idx]).float()
        dist_t = torch.from_numpy(self.dist[idx]).float()

        fidx = self.inverse_LUT[idx]
        base = self.bases[fidx]
        
        org = self.origins[idx].astype(int)//32
        ctx = base[org[0]:org[0]+5,org[1]:org[1]+5,org[2]:org[2]+5].reshape(1,5,5,5)
        ctx_t = torch.from_numpy(ctx).float()

        return tidx, gt_grid_t, dist_t, ctx_t

    def __len__(self):
        return self.N_leaf

class GridVoxelDataset(torch.utils.data.Dataset):
    def __init__(self, origin_fn, gt_fn, dist_fn, base_fn, shuffle=True, neighbors=True):
        super(GridVoxelDataset, self).__init__()
        self.origins = np.load(origin_fn).astype(int)
        sset = []
        if neighbors:
            vox_step = 32
        else:
            vox_step = 1
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    sset.append(np.array([i,j,k])*vox_step)
        feat_ptr = 0

        grids = {}
        self.grids_LUT = np.zeros((self.origins.shape[0], 8), dtype=int)
        for i,c in enumerate(self.origins):
            for j,step in enumerate(sset):
                triplet = tuple(c + step)
                if not triplet in grids:
                    grids[triplet] = feat_ptr
                    feat_ptr += 1
                self.grids_LUT[i,j] = grids[triplet]
        
        self.N_latent = feat_ptr
        self.gt_grid = np.load(gt_fn)
        self.dist = np.load(dist_fn)
        base = np.load(base_fn)
        self.base = np.pad(base, 2, mode='constant', constant_values=0)
        self.N_leaf = self.origins.shape[0]
        self.N = self.gt_grid.sum()
        print(f"A total number of {self.N} points")
        self.shuffle = shuffle
        bases = []
        for idx in range(self.N_leaf):
            org = self.origins[idx].astype(int)//32
            ctx = self.base[org[0]:org[0]+5,org[1]:org[1]+5,org[2]:org[2]+5].reshape(1,5,5,5)
            bases.append(ctx)
        self.bases = np.array(bases)
        print(f'A total number of {feat_ptr} grid points')

    def __getitem__(self, idx):
        # Random sample 
        if self.shuffle:
            magic_num = 2113
            idx = (idx * magic_num) % self.N_leaf
        nidx = np.array([idx], dtype=np.float32)
        tidx = torch.tensor(nidx).long()
        gt_grid_t = torch.from_numpy(self.gt_grid[idx]).float()
        dist_t = torch.from_numpy(self.dist[idx]).float()
        ctx_t = torch.from_numpy(self.bases[idx]).float()
        feat_indices_t = torch.from_numpy(self.grids_LUT[idx]).long()
        return tidx, gt_grid_t, dist_t, ctx_t, feat_indices_t
    
    def get_all(self):
        gt_grid_t = torch.from_numpy(self.gt_grid).float()
        dist_t = torch.from_numpy(self.dist).float()
        ctx_t = torch.from_numpy(self.bases).float()
        return gt_grid_t, dist_t, ctx_t, self.grids_LUT

    def __len__(self):
        return self.N_leaf

class LoadedVideoVoxelDataset(torch.utils.data.Dataset):
    def __init__(self, origin_fs, gt_fs, dist_fs):
        super(LoadedVideoVoxelDataset, self).__init__()
        self.origins_list = []
        self.gt_grid_list = []
        self.dist_list = []
        MAX_NLEAF = 2048
        self.inverse_LUT = np.zeros((len(origin_fs)*MAX_NLEAF), dtype=int)
        ptr = 0
        cnt = 0
        for origin_fn, gt_fn, dist_fn in zip(origin_fs, gt_fs, dist_fs):
            origin = np.load(origin_fn)
            gt = np.load(gt_fn)
            dist = np.load(dist_fn)
            assert origin.shape[0] == gt.shape[0]
            assert gt.shape[0] == dist.shape[0]
            self.origins_list.append(origin)
            self.gt_grid_list.append(gt)
            self.dist_list.append(dist)
            N = origin.shape[0]
            self.inverse_LUT[ptr:ptr+N] = cnt
            ptr = ptr + N
            cnt += 1

        self.origins = np.concatenate(self.origins_list, 0)
        self.gt_grid = np.concatenate(self.gt_grid_list, 0)
        self.dist = np.concatenate(self.dist_list, 0)

        self.N_leaf = self.origins.shape[0]
        self.N = self.gt_grid.sum()
        print(f"A total numbero of {self.N} points")

    def __getitem__(self, idx):
        # Random sample 
        nidx = np.array([idx], dtype=np.float32)
        tidx = torch.tensor(nidx).long()
        gt_grid_t = torch.from_numpy(self.gt_grid[idx]).float()
        dist_t = torch.from_numpy(self.dist[idx]).float()
        return tidx, gt_grid_t, dist_t
    
    def get_all(self):
        gt_grid_t = torch.from_numpy(self.gt_grid).float()
        dist_t = torch.from_numpy(self.dist).float()
        return gt_grid_t, dist_t

    def __len__(self):
        return self.N_leaf
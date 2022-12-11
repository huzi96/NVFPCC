# NVFPCC
Code repository for paper **Yueyu Hu, Yao Wang, "Learning Neural Volumetric Field for Point Cloud Geometry Compression", PCS 2022.**
## Note:
The code available here is still a messy version. It has not been fully tested and contains redundant code segments. Future updates will resolve these issues. If you have any concerns or encountered any problem, please feel free to contact me via email.

We would like to acknowledge the contribution of @NJUVISION at [NJUVISION/SparsePCGC](https://github.com/NJUVISION/SparsePCGC) in this project. They provide excellent examples in implementing voxel operations with MinkowskiEngine.

Note that the current implementation relies on some functionality in MinkowskiEngine. But this is not necessary and will possibly be removed in future updates. In order to use the current version, please refer to [link](https://github.com/NVIDIA/MinkowskiEngine) for guides to install MinkowskiEngine.

Please refer to [link](http://plenodb.jpeg.org/pc/8ilabs) to download ```longdress_vox10_1300.ply``` used in this example. Random seeds used in the code are available at [Google Drive](https://drive.google.com/drive/folders/1P0GB9Xn63-WChhturqxvGnuJFP-WuqQZ?usp=share_link).

## Usage
### 1. Build ground truth voxel occupancy grid and distance grid
#### a. Build octree and write out leaf node origins
First compile ```get_octree.cpp```,

> ```g++ get_octree.cpp -o get_octree -O2```

Run it to get origins and shallow subtree binary representation, 

> ```./get_octree longdress_vox10_1300.ply longdress_vox10_1300_l5_origins.txt longdress_vox10_1300_l5_subtree.txt```

#### b. With the origins, get the voxel grids at each leaf node,

> ```python util_get_grids.py longdress_vox10_1300.ply 5```

The script will produce,
* Origins: ```longdress_vox10_1300_l5_origins.npy```
* Occupancy grids: ```longdress_vox10_1300_l5_gt_grid.npy```
* Distance grids: ```longdress_vox10_1300_l5_dist.npy```

### 2. Train an NVF to represent the voxel grids
Example: train a model with longdress_vox10_1300.ply, saving checkpoints to directory ```ckpts```, using ```lambda=200```, with specified channel configurations.

> ```python NVFPCC.py train longdress_vox10_1300.ply --checkpoint_dir ckpts --batchsize 16 --lambda 200 --lr 1e-3  --w1 10 --w2 57 --wemb 5 --shuffle True --chanstr 8,16,8,8 --ch 3```

The script will produce checkpoints that is used later for testing.

### 3. Test encoding and decoding.
#### a. Quantize network weights
Quantize the weight we got at ```ckpts/0500.ckpt``` into 4 bits.

> ```python manipulate_weights.py ckpts/0500.ckpt 0500_quantized_q4.ckpt 16```

You may now check the number of bits needed to code the quantized checkpoint by,
> ```python util_code_quantized_weights.py 0500_quantized_q4.ckpt 0500_quantized_q4.bin```

#### b. Encode latent code and network weights

First build the arithmetic coder by,
> ```g++ module_arithmeticcoding.cpp -o module_arithmeticcoding -O2```

Then run the encoding script,

> ```python NVFPCC.py encode longdress_vox10_1300.ply --batchsize 1 --chanstr 8,16,8,8 --ch 3 --load_weights 0500_quantized_q4.ckpt --load_emb ckpts/0500_emb.ckpt --thh 0.65 --pack_fn pack.pk```

This command will do actual entropy coding and give the length of the bit-streams coding the network weights and the latent codes. The script will generate a pickle file, containing all the necessary information for decoding, including the bit-streams mentioned above. The script will also generate ```rc_enc.ply```, the reconstructed point cloud by the encoder.

#### c. Decode from the bit-stream

> ```python NVFPCC.py decode pack.pk --batchsize 1 --chanstr 8,16,8,8 --ch 3 --thh 0.64```

This command will decode from pack.pk and generate rc_dec.ply. One can verify that rc_enc.ply and rc_dec.ply are identical to each other.

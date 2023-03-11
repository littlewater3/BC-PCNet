import os, sys, glob, torch
# sys.path.append("../")
[sys.path.append(i) for i in ['.', '..']]
import numpy as np
import torch
import random
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from lib.benchmark_utils import to_o3d_pcd, to_tsfm, KDTree_corr
from lib.utils import load_obj

from lib.benchmark_utils import to_o3d_pcd, to_tsfm, get_correspondences

#3dmatch数据预处理
class _3DMatch(Dataset):

    def __init__(self, config,split, data_augmentation=True):
        super(_3DMatch, self).__init__()

        assert split in ['train','val','test']   #断言语句

        if 'overfit' in config.exp_dir:
            d_slice = config.batch_size
        else :
            d_slice = None

        self.infos = self.read_entries( config.split[split] , config.data_root, d_slice=d_slice )


        self.base_dir = config.data_root
        self.data_augmentation = data_augmentation
        self.config = config

        self.rot_factor = 1.
        self.augment_noise = config.augment_noise
        self.max_points = 30000

        self.overlap_radius = 0.0375



    def read_entries (self, split, data_root, d_slice=None, shuffle= True):    #加载pkl文件
        infos = load_obj(split)  # we use the split prepared by Predator   采用Predator的数据分类
        if d_slice:
            for  k, v  in  infos.items():
                infos[k] = v[:d_slice]
        return infos




    def __len__(self):       #通过内置函数len，可以读取到有数据个数   （从PKL文件里读取多少行）
        return len(self.infos['rot'])

    def __getitem__(self, item, debug=False):      #通过getitem内在函数读取数据，item为索引，需要将读取的数据处理为统一格式
        # get transformation
        rot = self.infos['rot'][item]
        trans = self.infos['trans'][item]
        if 'gt_cov' in self.infos:
            gt_cov = self.infos['gt_cov'][item]
        else :
            gt_cov = None

        # get pointcloud
        print(self.infos['src'][item])
        print(self.infos['tgt'][item])
        src_path = os.path.join(self.base_dir, self.infos['src'][item])    #用于拼接地址  原点云
        tgt_path = os.path.join(self.base_dir, self.infos['tgt'][item])    #目标点云
        src_pcd = torch.load(src_path)
        #print(type(src_pcd))        N*3 的数组
        #print(src_pcd.shape)
        tgt_pcd = torch.load(tgt_path)

        # if we get too many points, we do some downsampling    因点太多，而缩减采样
        if (src_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(src_pcd.shape[0])[:self.max_points]   #按行数随机排列，取最大点数的行，取出的是索引
            src_pcd = src_pcd[idx]
        if (tgt_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(tgt_pcd.shape[0])[:self.max_points]
            tgt_pcd = tgt_pcd[idx]


        if debug:
            import mayavi.mlab as mlab
            c_red = (224. / 255., 0 / 255., 125 / 255.)
            c_pink = (224. / 255., 75. / 255., 232. / 255.)
            c_blue = (0. / 255., 0. / 255., 255. / 255.)
            scale_factor = 0.02
            # mlab.points3d(s_pc[ :, 0]  , s_pc[ :, 1],  s_pc[:,  2],  scale_factor=scale_factor , color=c_blue)
            mlab.points3d(src_pcd[ :, 0] , src_pcd[ :, 1], src_pcd[:,  2], scale_factor=scale_factor , color=c_red)
            mlab.points3d(tgt_pcd[ :, 0] , tgt_pcd[ :, 1], tgt_pcd[:,  2], scale_factor=scale_factor , color=c_blue)
            mlab.show()



        # add gaussian noise
        if self.data_augmentation:     #数据增强en1
            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2 / self.rot_factor  # anglez, angley, anglex  #欧拉角
            #print(euler_ab)
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()  #转化为数组
            #print(rot_ab)
            if (np.random.rand(1)[0] > 0.5):     #随机数，0.5的概率
                src_pcd = np.matmul(rot_ab, src_pcd.T).T     #变换src
                rot = np.matmul(rot, rot_ab.T)
            else:
                tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T    #变换tgt
                rot = np.matmul(rot_ab, rot)
                trans = np.matmul(rot_ab, trans)

            src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.augment_noise   #增加高斯噪声
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * self.augment_noise

        # get correspondence at fine level
        tsfm = to_tsfm(rot, trans)
        correspondences = get_correspondences(to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd), tsfm,self.overlap_radius)
        #print(correspondences.shape)
        #print(type(correspondences))
        #获取correspondences对应关系


        if debug:
            import mayavi.mlab as mlab
            c_red = (224. / 255., 0 / 255., 125 / 255.)
            c_pink = (224. / 255., 75. / 255., 232. / 255.)
            c_blue = (0. / 255., 0. / 255., 255. / 255.)
            scale_factor = 0.02
            # mlab.points3d(s_pc[ :, 0]  , s_pc[ :, 1],  s_pc[:,  2],  scale_factor=scale_factor , color=c_blue)
            mlab.points3d(src_pcd[ :, 0] , src_pcd[ :, 1], src_pcd[:,  2], scale_factor=scale_factor , color=c_red)
            mlab.points3d(tgt_pcd[ :, 0] , tgt_pcd[ :, 1], tgt_pcd[:,  2], scale_factor=scale_factor , color=c_blue)
            mlab.show()


        if (trans.ndim == 1):
            trans = trans[:, None]


        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)    #取点云数据的行数并将元素置为1，n*1
        #print(src_feats.shape)    #19082*1
        #print(src_feats)
        #print(type(src_feats))
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        #print(tgt_feats.shape)   #18977*1
        rot = rot.astype(np.float32)  #3*3
        #print(rot)        
        trans = trans.astype(np.float32)    #3*1
        #print(trans)

        return src_pcd, tgt_pcd, src_feats, tgt_feats, correspondences, rot, trans, gt_cov     #返回值，



if __name__ == '__main__':
    from lib.utils import load_config
    from easydict import EasyDict as edict
    from lib.tictok import Timers
    import yaml
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return '_'.join([str(i) for i in seq])
    yaml.add_constructor('!join', join)

    config = "configs/train/3dmatch.yaml"
    with open(config,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config = edict(config)
    config.timers=Timers()
    D = _3DMatch(config, "train")
    #D = _3DMatch(config, "test")
    #print(len(D))
    for i in range (len(D)):

        try:
            if i%1000 == 0 :
                print (i,"/",len(D))
            D.__getitem__(i, debug=True)
        except:
            pass
        #     print ( D.data_entries[i] )
        #     print (os.remove(D.data_entries[i]) )


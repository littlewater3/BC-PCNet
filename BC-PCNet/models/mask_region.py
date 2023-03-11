import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import mayavi.mlab as mlab
import time
from numpy import *
from scipy.spatial.transform import Rotation
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms

#颜色
c_red = (224. / 255., 0 / 255., 125 / 255.)
c_pink = (224. / 255., 75. / 255., 232. / 255.)
c_blue = (0. / 255., 0. / 255., 255. / 255.)
c_green = (0. / 255., 255. / 255., 0. / 255.)
c_gray1 = (100. / 255., 100. / 255., 100. / 255.)
c_gray2 = (175. / 255., 175. / 255., 175. / 255.)
c_yellow = (225. / 255., 255 / 255., 0 / 255.)



class RegionMask(nn.Module):
    def  __init__(self,config):
        super(RegionMask, self).__init__()    #对父类初始化
        
        self.d_model = config['feature_dim']

        self.m_1 = nn.Linear(self.d_model,64)
        self.m_2 = nn.Linear(64,64)  #64
        self.m_3 = nn.Linear(64,2)

 
    def forward(self,src_feat,tgt_feat):      

        src_feat = self.m_1(src_feat)
        src_feat = F.relu(src_feat)    
        src_feat = self.m_2(src_feat)
        src_feat = F.relu(src_feat)   
        src_feat = self.m_3(src_feat)
        src_feat_mask = F.relu(src_feat)   

        tgt_feat = self.m_1(tgt_feat)
        tgt_feat = F.relu(tgt_feat)    
        tgt_feat = self.m_2(tgt_feat)
        tgt_feat = F.relu(tgt_feat)   
        tgt_feat = self.m_3(tgt_feat)
        tgt_feat_mask = F.relu(tgt_feat)   


        return src_feat_mask.squeeze(0),tgt_feat_mask.squeeze(0)



#最小临
def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src   对于在src中的每个点，在dst中找最近邻
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor     输出最小零的欧氏距离
        indices: dst indices of the nearest neighbor      dst的最小临索引
    '''

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

#采用mlab单独点云显示，需要矩阵格式
def show_pointcloud(src,scale_factor=0.05):
    mlab.points3d(src[:, 0], src[:, 1], src[:, 2],scale_factor=scale_factor * 0.25,color=c_blue)
    mlab.show()

#采用mlab显示两个点云
def show_double_pointcloud(src,tgt,scale_factor=0.05):
    mlab.points3d(src[:, 0], src[:, 1], src[:, 2],scale_factor=scale_factor * 0.25,color=c_blue)
    mlab.points3d(tgt[:, 0], tgt[:, 1], tgt[:, 2],scale_factor=scale_factor * 0.25,color=c_green)
    mlab.show()

#显示多个点云
def show_mul_pointclud(src,tgt,mask_1,mask_2,scale_factor=0.05):
    mlab.points3d(src[:, 0], src[:, 1], src[:, 2],scale_factor=scale_factor * 0.25,color=c_blue)
    mlab.points3d(tgt[:, 0], tgt[:, 1], tgt[:, 2],scale_factor=scale_factor * 0.25,color=c_green)
    mlab.points3d(mask_1[:, 0], mask_1[:, 1], mask_1[:, 2],scale_factor=scale_factor * 0.25,color=c_pink)
    mlab.points3d(mask_2[:, 0], mask_2[:, 1], mask_2[:, 2],scale_factor=scale_factor * 0.25,color=c_yellow)
    mlab.show()


#open3d读取并输出点云数据
def read_cd():
    # 从文件中读取点云，o3d格式
    pcd = o3d.io.read_point_cloud('./data/cloud_bin_1.ply')
    print(pcd)  #点云数
    #print(type(pcd))
    print(np.asarray(pcd.points)) #点云的坐标矩阵
    print(type(pcd.points))
    o3d.visualization.draw_geometries([pcd],
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])

#将o3d数据转化为array
def to_xyz(pcd):
    xyz = np.asarray(pcd.points)
    return xyz

#pth文件读取
def readpth(path_1=None,path_2=None):
    src_pcd = torch.load('./data/cloud_bin_'+path_1+'.pth').astype(np.float32)
    tgt_pcd = torch.load('./data/cloud_bin_'+path_2+'.pth').astype(np.float32)
    #print(src_pcd)
    #print(tgt_pcd)
    return src_pcd,tgt_pcd

#txt位姿矩阵数据读取
def read_txt(path):
    A = zeros((4, 4), dtype=float)
    f = open('./data/cloud_bin_'+path+'.info.txt')
    lines = f.readlines() 
    del lines[0]    #删除指定的行
    A_row = 0
    for line in lines:  # 把lines中的数据逐行读取出来    一个line读取的是一行的数据
        #去除  \n:换行   \t:tab   ' ':空格
        line = line.strip('')  #去除空格
        list = line.split('\t')  # 将tap作为小的分界
        A[A_row:] = list[0:4]  # 把处理后的数据放到方阵A中。list[0:4]表示列表的0,1,2,3列数据放到矩阵A中的A_row行
        A_row += 1  # 然后方阵A的下一行接着读
    #print(A)
    return A

def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(to_array(xyz))    #张量转换为数组
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

'''
def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor
'''

#相对矩阵求解（暂未使用）
def relative_matrix():
    A = zeros((4, 4), dtype=float)
    return A

#合并旋转矩阵和位移矩阵
def to_tsfm(rot,trans):
    tsfm = np.eye(4)    #创建个4维的对角矩阵
    tsfm[:3,:3]=rot   #3*3
    tsfm[:3,3]=trans.flatten()   #1*2
    return tsfm     #4*4

#最小临的索引
def get_nb(src_xyz,tgt_xyz):
    distance,index=nearest_neighbor(tgt_xyz,src_xyz)
    return distance,index

#根据阈值，将
def mask_nb(dis,index,radius):
    mask = np.zeros((len(dis),1))
    for i in range(len(dis)):
        if dis[i] <= radius:
            mask[i] = 1
        else:
            mask[i] = 0
    #print(mask.shape)
    #print(mask)
    #savepath = "C:\\Users\\TT\\Desktop\\test_PointCloud\\"+ "mask.txt"  #存储mask文件
    #np.savetxt(savepath,mask)
    return mask

#读取标记了mask的点，输出为数组
def read_mask_point(mask,xyz):
    mask_xyz = np.empty(shape=(100000,3))
    #print(mask.shape[0])
    j = 0
    for i in range(mask.shape[0]):
        if mask[i] == 1:
            mask_xyz[j] = xyz[i]
            j = j+1
    mask_xyz = mask_xyz[0:j]
    return mask_xyz

def mask_region(s_pcd,t_pcd,mask_radius,data):
    s_pcd = s_pcd.squeeze(0)
    t_pcd = t_pcd.squeeze(0)
    s_pcd = (torch.matmul( data['batched_rot'][0], s_pcd.T ) + data['batched_trn'][0]).T
    src_pcd = s_pcd.cpu().numpy()
    tgt_pcd = t_pcd.cpu().numpy()
    dis_src,index_src = get_nb(src_pcd,tgt_pcd)
    mask_tgt = mask_nb(dis_src,index_src,mask_radius)
    #mask_xyz_src = read_mask_point(mask_tgt,tgt_pcd)
    dis_tgt,index_tgt = get_nb(tgt_pcd,src_pcd)
    mask_src = mask_nb(dis_tgt,index_tgt,mask_radius)
    #mask_xyz_tgt = read_mask_point(mask_src,src_pcd)
    #show_mul_pointclud(src_pcd,tgt_pcd,mask_xyz_src,mask_xyz_tgt,scale_factor = 0.2)

    return torch.from_numpy(mask_src).squeeze(1).long(),torch.from_numpy(mask_tgt).squeeze(1).long()



if __name__ == '__main__':
    src,tgt = readpth('0','3')   #array格式
    #show_pointcloud(src)
    #show_double_pointcloud(src,tgt)
    src_pose = read_txt('0')    #pose
    tgt_pose = read_txt('3')
    src_pcd = to_o3d_pcd(src)   #转open3d格式
    tgt_pcd = to_o3d_pcd(tgt)
    src_pcd.transform(src_pose)    #位姿变换
    tgt_pcd.transform(tgt_pose)
    src_xyz = to_xyz(src_pcd)    #array格式
    tgt_xyz = to_xyz(tgt_pcd)
    #show_pointcloud(src_xyz)
    #show_double_pointcloud(src_xyz,tgt_xyz)    

    mask_radius = 0.032   #粗0.032   精0.026
    dis_1,index_1 = get_nb(src_xyz,tgt_xyz)
    mask_1 = mask_nb(dis_1,index_1,mask_radius)
    mask_xyz_1 = read_mask_point(mask_1,tgt_xyz)
    dis_2,index_2 = get_nb(tgt_xyz,src_xyz)    
    mask_2 = mask_nb(dis_2,index_2,mask_radius)
    mask_xyz_2 = read_mask_point(mask_2,src_xyz)
    show_mul_pointclud(src_xyz,tgt_xyz,mask_xyz_1,mask_xyz_2)



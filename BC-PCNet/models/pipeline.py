from models.blocks import *
from models.backbone import KPFCN
from models.transformer import RepositioningTransformer
from models.matching import Matching
from models.procrustes import SoftProcrustesLayer
from models.mask_region import mask_region
from models.mask_region import RegionMask
from models.iterative_TMP import IterativeOptimal
from models.Last_TMP import lastTMP

class Pipeline(nn.Module):

    def __init__(self, config):
        super(Pipeline, self).__init__()
        self.config = config
        self.backbone = KPFCN(config['kpfcn_config'])       #KPFCN框架
        self.pe_type = config['coarse_transformer']['pe_type']    #pe_type模式选择
        self.positioning_type = config['coarse_transformer']['positioning_type']    #定位方式
        self.coarse_transformer = RepositioningTransformer(config['coarse_transformer'])     #self + cross + position
        self.iterative_optimal = IterativeOptimal(config['iterative_optimal'])   #迭代优化，self + cross + position
        self.last_TMP = lastTMP(config['Last_TMP'])    #最后一层TMP，需要多加一层  self + cross + position + self + cross
        self.iter = config['iterative_optimal']['iter']    #迭代配置文件，只需要迭代层数
        self.region_mask = RegionMask(config['mask_region'])
        self.coarse_matching = Matching(config['coarse_matching'])
        self.soft_procrustes = SoftProcrustesLayer(config['coarse_transformer']['procrustes'])
        


    def forward(self, data,  timers=None):      #在train的时候运行forward

        self.timers = timers
        data.update({"src_feat_mask":{}})
        data.update({"tgt_feat_mask":{}})

        #kpfcv框架提取特征，原型Kpconv模型  
        if self.timers: self.timers.tic('kpfcn backbone encode')   #单次计时开始
        coarse_feats = self.backbone(data, phase="coarse")       #输入数据，得到提取的粗特征coarse_feats
        if self.timers: self.timers.toc('kpfcn backbone encode')   #结束本次计时，返回值为时间差toc

        if self.timers: self.timers.tic('coarse_preprocess')
        src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask = self.split_feats (coarse_feats, data)    #分离参数
        data.update({ 's_pcd': s_pcd, 't_pcd': t_pcd })    #更新原点云和目标点云   合并字典，增加s_pcd和t_pcb
        if self.timers: self.timers.toc('coarse_preprocess')

        #mask ground-truth
        if self.timers: self.timers.tic('mask_region')
        src_mask_region, tgt_mask_region = mask_region (s_pcd, t_pcd,self.config['mask_region']['mask_radius'],data)    #返回区域mask标签 ground-truth
        data.update({ 'src_mask_region': src_mask_region.cuda(), 'tgt_mask_region': tgt_mask_region.cuda() })    
        if self.timers: self.timers.toc('mask_region')

        #第一层
        if self.timers: self.timers.tic('coarse feature transformer')   
        src_feats, tgt_feats, src_pe, tgt_pe = self.coarse_transformer(src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask, data, timers=timers)    
        src_feat_mask , tgt_feat_mask = self.region_mask(src_feats,tgt_feats)   #非真实值，将特征变化维度
        data["src_feat_mask"][0] = src_feat_mask
        data["tgt_feat_mask"][0] = tgt_feat_mask
        #print('执行第0层TMP')
        if self.timers: self.timers.toc('coarse feature transformer')



        #迭代优化   iter预设值为2
        if self.timers: self.timers.tic('iterative optimal')   #迭代优化
        for i in range(self.iter):
            src_feats, tgt_feats, src_pe, tgt_pe = self.iterative_optimal(src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask, data, timers=timers)
            src_feat_mask , tgt_feat_mask = self.region_mask(src_feats,tgt_feats)
            data["src_feat_mask"][i+1] = src_feat_mask
            data["tgt_feat_mask"][i+1] = tgt_feat_mask
            i = i+1
            #print("执行第"+str(i)+"层TMP")
        if self.timers: self.timers.toc('iterative optimal')


        #最后一层增加self+cross
        if self.timers: self.timers.tic('last TMP') 
        src_feats, tgt_feats, src_pe, tgt_pe = self.last_TMP(src_feats, tgt_feats, s_pcd, t_pcd, src_mask, tgt_mask, data, timers=timers)
        src_feat_mask , tgt_feat_mask = self.region_mask(src_feats,tgt_feats)
        data["src_feat_mask"][self.iter+1] = src_feat_mask
        data["tgt_feat_mask"][self.iter+1] = tgt_feat_mask
        #print('执行第'+str(self.iter+1)+'层TMP')
        if self.timers: self.timers.toc('last TMP')


        if self.timers: self.timers.tic('match feature coarse')    #单独特征匹配单元
        conf_matrix_pred, coarse_match_pred = self.coarse_matching(src_feats, tgt_feats, src_pe, tgt_pe, src_mask, tgt_mask, data, pe_type = self.pe_type)
        data.update({'conf_matrix_pred': conf_matrix_pred, 'coarse_match_pred': coarse_match_pred })
        if self.timers: self.timers.toc('match feature coarse')

        if self.timers: self.timers.tic('procrustes_layer')   #解出R,T
        R, t, _, _, _, _ = self.soft_procrustes(conf_matrix_pred, s_pcd, t_pcd, src_mask, tgt_mask)
        data.update({"R_s2t_pred": R, "t_s2t_pred": t})
        if self.timers: self.timers.toc('procrustes_layer')

        return data



    #split_feats用于将feature map尺寸处理一下
    def split_feats(self, geo_feats, data):

        pcd = data['points'][self.config['kpfcn_config']['coarse_level']]   #倒数第二层的点

        src_mask = data['src_mask']
        tgt_mask = data['tgt_mask']
        src_ind_coarse_split = data[ 'src_ind_coarse_split']    #区分点云数据索引
        tgt_ind_coarse_split = data['tgt_ind_coarse_split']
        src_ind_coarse = data['src_ind_coarse']
        tgt_ind_coarse = data['tgt_ind_coarse']

        b_size, src_pts_max = src_mask.shape
        tgt_pts_max = tgt_mask.shape[1]

        src_feats = torch.zeros([b_size * src_pts_max, geo_feats.shape[-1]]).type_as(geo_feats)
        tgt_feats = torch.zeros([b_size * tgt_pts_max, geo_feats.shape[-1]]).type_as(geo_feats)
        src_pcd = torch.zeros([b_size * src_pts_max, 3]).type_as(pcd)
        tgt_pcd = torch.zeros([b_size * tgt_pts_max, 3]).type_as(pcd)

        src_feats[src_ind_coarse_split] = geo_feats[src_ind_coarse]
        tgt_feats[tgt_ind_coarse_split] = geo_feats[tgt_ind_coarse]
        src_pcd[src_ind_coarse_split] = pcd[src_ind_coarse]
        tgt_pcd[tgt_ind_coarse_split] = pcd[tgt_ind_coarse]

        return src_feats.view( b_size , src_pts_max , -1), \
               tgt_feats.view( b_size , tgt_pts_max , -1), \
               src_pcd.view( b_size , src_pts_max , -1), \
               tgt_pcd.view( b_size , tgt_pts_max , -1), \
               src_mask, \
               tgt_mask













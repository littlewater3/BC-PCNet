import os, torch, json, argparse, shutil
from easydict import EasyDict as edict
import yaml
from datasets.dataloader import get_dataloader, get_datasets
from models.pipeline import Pipeline
from lib.utils import setup_seed
from lib.tester import get_trainer
from models.loss import MatchMotionLoss
from lib.tictok import Timers
from configs.models import architectures

from torch import optim


setup_seed(0)     #生成随机种子

def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])

yaml.add_constructor('!join', join)    ## 用add_constructor方法为指定yaml标签添加构造器


if __name__ == '__main__':
    # load configs 加载配置文件
    parser = argparse.ArgumentParser()   #建立解析器
    parser.add_argument('--config', type=str, default="configs/train/3dmatch.yaml", help= 'Path to the config file.')    #添加参数文件，读取yaml文件
    parser.add_argument("--local_rank", type=int, default=1, help="number of cpu threads to use during batch generation")
    args = parser.parse_args()      #解析参数文件
    with open(args.config,'r') as f:      #读取参数文件
        config = yaml.load(f, Loader=yaml.Loader)

    #设置一套模式
    config['snapshot_dir'] = 'snapshot/%s/%s' % (config['dataset']+config['folder'], config['exp_dir'])
    config['tboard_dir'] = 'snapshot/%s/%s/tensorboard' % (config['dataset']+config['folder'], config['exp_dir'])
    config['save_dir'] = 'snapshot/%s/%s/checkpoints' % (config['dataset']+config['folder'], config['exp_dir'])
    config = edict(config)

    #创建上述目录  
    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)

    if config.gpu_mode:
        config.device = torch.device("cuda:0")
    else:
        config.device = torch.device('cpu')
    
    # backup the
    if config.mode == 'train':
        #print({config.snapshot_dir})     #创建文件夹，
        '''
        print(os.path.exists(config['snapshot_dir']))
        #os.path.exists(models)
        print(config['snapshot_dir'])
        os.system(f'cp -r models config.snapshot_dir')
        os.system(f'cp -r configs {config.snapshot_dir}')
        os.system(f'cp -r cpp_wrappers {config.snapshot_dir}')
        os.system(f'cp -r datasets {config.snapshot_dir}')
        os.system(f'cp -r kernels {config.snapshot_dir}')
        os.system(f'cp -r lib {config.snapshot_dir}')
        shutil.copy2('main.py',config.snapshot_dir)
        '''
    
    # model initialization    模型初始化
    config.kpfcn_config.architecture = architectures[config.dataset]  #确定数据集种类
    config.model = Pipeline(config)   #载入模型  框架主体
    #pipeline导入模型框架
    #config.model = KPFCNN(config)

    # create optimizer    载入优化器：梯度下降的方法（斜率大小）
    if config.optimizer == 'SGD':   #选择SGD优化
        config.optimizer = optim.SGD(
            config.model.parameters(), 
            lr=config.lr,      #初始学习率0.015
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            )
    elif config.optimizer == 'ADAM':   #非凸优化
        config.optimizer = optim.Adam(
            config.model.parameters(), 
            lr=config.lr,      #初始学习率
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    

    #create learning rate scheduler   学习速率的选择（步长减小）
    if  'overfit' in config.exp_dir :   #超出部分
        config.scheduler = optim.lr_scheduler.MultiStepLR(     #多阶梯学习率
            config.optimizer,
            milestones=[config.max_epoch-1], # fix lr during overfitting    混合学习速率
            gamma=0.1,     #每次乘0.1来减小
            last_epoch=-1)

    else:
        config.scheduler = optim.lr_scheduler.ExponentialLR(      #指数学习速率
            config.optimizer,
            gamma=config.scheduler_gamma,
        )


    config.timers = Timers()

    # create dataset and dataloader
    train_set, val_set, test_set = get_datasets(config)    #get_dataset用于对单一数据的读取，返回值为对象
    config.train_loader, neighborhood_limits = get_dataloader(train_set,config,shuffle=True)  #get_dataloader用于对单一数据拼接乘一组batchsise
    config.val_loader, _ = get_dataloader(val_set, config, shuffle=False, neighborhood_limits=neighborhood_limits)
    config.test_loader, _ = get_dataloader(test_set, config, shuffle=False, neighborhood_limits=neighborhood_limits)
    
    # config.desc_loss = MetricLoss(config)
    config.desc_loss = MatchMotionLoss (config['train_loss'])     #损失函数

    trainer = get_trainer(config)
    if(config.mode=='train'):
        trainer.train()
    else:
        trainer.test()

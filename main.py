import argparse
import os
from solver import Solver
from data_loader import *
#from data_loader import *
from torch.backends import cudnn
from torch.utils import data
from torchvision import transforms as T

def main(config):
    cudnn.benchmark = True

    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path )
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    if config.train_dataset == 'african':
        config.img_size = (640, 640)
    elif config.train_dataset == 'asian':
        config.img_size = (640, 480)
    elif config.train_dataset == 'mobile':
        config.img_size = (384, 384)

    print(config)

    train_loader=data.DataLoader(Train_dataset(root=config.train_path, dataset_type=config.train_dataset,img_size = config.img_size ,transform=traindata_augmentation,mode='train'),
                                batch_size=config.batch_size,
                                shuffle=True,
                                num_workers=config.num_workers)
    valid_loader=data.DataLoader(Train_valid_dataset(root=config.valid_path, dataset_type=config.train_dataset,img_size = config.img_size ,transform=testdata_augmentation,mode='valid'),
                                batch_size=config.batch_size,
                                shuffle=True,
                                num_workers=config.num_workers)
    if config.test_mode == 1:
        test1_loader=data.DataLoader(Test1_dataset(root=config.test_path, dataset_type=config.train_dataset,img_size = config.img_size ,transform=testdata_augmentation,mode='test'),
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=config.num_workers)
    elif config.test_mode == 2:
        test2_loader=data.DataLoader(Test2_dataset(root=config.test_path, dataset_type=config.train_dataset,img_size = config.img_size ,transform=testdata_augmentation,mode='test'),
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=config.num_workers)

    if config.test_mode == 1:
        solver = Solver(config, train_loader, valid_loader, test1_loader)
    elif config.test_mode == 2:
        solver = Solver(config, train_loader, valid_loader, test2_loader)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test' and config.test_mode == 1:
        solver.test_1()
    elif config.mode == 'test' and config.test_mode == 2:
        solver.test_2()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0002)
    
    parser.add_argument('--beta1', type=float, default=0.9)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='./models/african_best.pth')
    parser.add_argument('--img_size', type=tuple, default=(640, 640))
    parser.add_argument('--train_path', type=str, default='./dataset/train/')
    parser.add_argument('--valid_path', type=str, default='./dataset/valid/')
    parser.add_argument('--test_path', type=str, default='./dataset/test/')
    parser.add_argument('--test_mode', type=int, default=1, help='1 or 2')#若test_mode==1，则test时会计算评估指标。若==2，则不计算评估指标。
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--train_dataset', type=str, default='african', help='choose train datasets, african, asian of mobile')

    config = parser.parse_args()
    main(config)

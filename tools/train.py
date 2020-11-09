'''

    create in 2020.11.09

'''

import os
import sys
sys.path.append('..')
import yaml
import argparse


from utils.fileu import *
from pse_label import *
from easydict import EasyDict




def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model', type=str, default='OpenPCDet', help='specify model')
    parser.add_argument('--num_iters', type=int, default=10, help='ssl run iteration')


    args = parser.parse_args()



    with open('../models/'+args.model+'.yaml') as f:
        cfg = yaml.load(f)
        cfg = EasyDict(cfg)


    return args, cfg


def train_model(cfg):
    os.chdir(cfg.path)
    os.system(cfg.train_cmd)




def ssl_main():
    args, cfg = parse_config()


    for i in range(args.num_iters):
        # 迭代训练模型
        train_model(cfg)

        # 过滤并生成伪标签
        filter_label(cfg)







if __name__ == "__main__":
    ssl_main()














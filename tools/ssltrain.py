import sys
# add model path
sys.path.append('')


import os 
import yaml
import shutil

from utils.label_utils import get_kitti_label
from datasets.ssldataset import KittiDataset



def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def filter_label(cfgs):
    raw_path = cfgs.raw_label_path
    pseudo_path = cfgs.pseudo_label_path
    filter_thresh = cfgs.thresh

    check_path(raw_path)
    check_path(pseudo_path)

    filenames = os.listdir(label_path)
    
    
    for name in filenames:
        with open(os.path.join(raw_path, name)) as f:
            lines = f.readlines()

        # 过滤标签
        line_tmp = []
        for line in lines:
            obj = get_kitti_label(line)
            if obj[-1] > filter_thresh:
                line_tmp.append(line)

        # 大于阈值的写入
        if line_tmp != []:
            with open(os.path.join(pseudo_path, name)) as f:
                for line in line:
                    f.write(line[:-1] + '\n')
            


def ssl_main():
    epochs = 10
    with open('cfgs/ssl.yaml', 'r') as f:
        cfgs = yaml.load(f)


    # create first pseudo label
    os.system('./scripts/eval.sh')
    filter_label(cfgs)

    for epoch in range(epochs):
        # train
        os.system('./scripts/train.sh')

        # get pseudo label and filter
        os.system('./scripts/eval.sh')
        filter_label(cfgs)




def test_dataloader():
    class_names = ['Car', 'Pedestrian']
    datapath = '/data/kitti/training'

    with open('cfgs/dataset.yaml', 'r') as f:
        dataset_cfg = yaml.load(f)

    dataset = KittiDataset(dataset_cfg, class_names, datapath, training=True)
    # dataloader = DataLoader(
    #     dataset, batch_size=batch_size, pin_memory=True, num_workers=0,
    #     shuffle=True, collate_fn=dataset.collate_batch,
    #     drop_last=False, sampler=sampler, timeout=0
    # )
    
    # for i, data_dict in enumerate(dataloader):
    #     print(i, data_dict.keys())



if __name__ == "__main__":
    # ssl_main()
    test_dataloader()

            










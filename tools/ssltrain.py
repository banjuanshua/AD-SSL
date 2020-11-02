import os 
import shutil

from utils.label_utils import get_kitti_label



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
    cfgs = None

    # create first pseudo label
    os.system('./scripts/eval.sh')
    filter_label(cfgs)

    for epoch in range(epochs):
        # train
        os.system('./scripts/train.sh')

        # get pseudo label and filter
        os.system('./scripts/eval.sh')
        filter_label(cfgs)




if __name__ == "__main__":
    ssl_main()

            










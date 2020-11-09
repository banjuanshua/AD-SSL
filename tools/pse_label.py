import os
import shutil



def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)


def filter_label(cfg):
    eval_path = cfg.eval_label_path
    pseudo_path = cfg.pseudo_label_path
    train_path = cfg.train_label_path

    # 清空并创建目录
    check_path(eval_path)
    check_path(pseudo_path)
    check_path(train_path)

    # 过滤伪标签
    filenames = os.listdir(eval_path)
    for filename in filenames:
        file_eval = os.path.join(eval_path, filename)
        file_pseudo = os.path.join(pseudo_path, filename)

        f1 = open(file_eval, 'r')
        f2 =  open(file_pseudo, 'r')
        lines = f1.readlines()

        for line in lines:
            # todo  过滤策略

            f2.write()

    # todo test
    shutil.copy(pseudo_path, train_path)


if __name__ == '__main__':
    filter_label()

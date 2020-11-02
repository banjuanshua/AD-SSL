import os



# todo test
def get_kitti_label(line):
    line = line.split(' ')
    obj = [line[0]]
    line = [float(x) for x in line[1:]]
    obj.extend(line)
    return obj
    

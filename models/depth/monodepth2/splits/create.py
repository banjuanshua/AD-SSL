
filename_std = 'eigen_zhou/std_train_files.txt'
filename = 'eigen_zhou/train_files.txt'

f_std = open(filename_std, 'r')
f = open(filename, 'w')

for line in f_std.readlines():
    print(line)
    if '2011_09_26_drive_0001_sync' in line:
        f.write(line)


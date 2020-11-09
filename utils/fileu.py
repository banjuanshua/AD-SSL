

def change_file(filepath, line_number, flag):
    '''

    :param filepath:
    :param line_number:
    :param flag:  0： 注释掉    1：取消注释
    :return:
    '''



    newlines = ''


    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if (i+1) == line_number:
                if flag == 0:
                    line = '#' + line
                elif flag == 1:
                    line = line[1:]
            newlines += line

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(newlines)



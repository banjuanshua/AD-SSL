import os
import cv2


def carete_split_file(video_name):
    path = 'datasets/custom_data/'
    video_path = path + video_name
    folder_path = video_path.split('.')[0]
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    split_path = 'splits/'+video_name.split('.')[0]
    if not os.path.exists(split_path)
        os.mkdir(split_path)

    img_folder = video_path.split('.')[0] + '/'
    
    file_path = os.path.join(split_path, 'train_files.txt')
    f = open(file_path, 'w')
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    
    cnt = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        img_name = str(cnt) + '.jpg'
        img_path = img_folder + img_name
        print(img_path)
        cnt += 1
        cv2.imwrite(img_path, frame)
        txt_str = video_name + '/' + img_name
        f.write(img_path+' l\n')


if __name__ == '__main__':
    video_name = 'project_video.mp4'
    carete_split_file(video_name)
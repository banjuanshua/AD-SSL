import cv2
import numpy as np

def read_video(path):
    videp_path = '../output.mp4'
    cap = cv2.VideoCapture(videp_path)
    return cap


def read_img_xy(img):

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            print(x, y)
            cv2.imshow("image", img)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    while (1):
        cv2.imshow("image", img)
        if cv2.waitKey(0) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


def extract_orb(img):
    orb = cv2.ORB_create()
    # detection
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8),
                                  3000,
                                  qualityLevel=0.01,
                                  minDistance=7)

    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
    kps, des = orb.compute(img, kps)
    img = cv2.drawKeypoints(img, kps, None, color=(0, 255, 0), flags=0)
    return img, np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des


import numpy as np
import matplotlib.pyplot as plt

# 数学几何

class Point3d():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        print_text = f'x:{self.x} y:{self.y} z:{self.z}'
        return print_text

def image2sapce(x_arr, y_arr, depth, K):
    points3d = []

    u_arr = []
    v_arr = []
    z_arr = []

    for x, y in zip(x_arr, y_arr):
        z = depth[y][x]
        p2d = np.array([x, y, 1], np.float32)
        p2d *= z

        u_arr.append(p2d[0])
        v_arr.append(p2d[1])
        z_arr.append(z)

        invert_K = np.linalg.pinv(K)
        p3d = np.dot(invert_K, p2d)
        if p3d[2] < 100:
            p3d = Point3d(p3d[0], p3d[1], p3d[2])
            points3d.append(p3d)



    # uv = np.array([x_arr, y_arr, [1 for i in range(len(x_arr))]])

    # uz = np.array([u_arr, v_arr, z_arr])
    # invert_K = np.linalg.pinv(K)
    # p3d = np.dot(invert_K, uz)
    # print(p3d)
    # print(p3d.shape)
    #
    #
    # print('------------')
    #
    # print([el.x for el in points3d])
    # print([el.y for el in points3d])
    # sss

    return points3d


def fit_poly2d(x_arr, y_arr, coeff=2):
    return np.polyfit(y_arr, x_arr, coeff)


def fit_poly3d(points3d, coeff=2):
    x_arr = [el.x for el in points3d]
    z_arr = [el.z for el in points3d]
    poly = np.polyfit(z_arr, x_arr, coeff)
    return poly




if __name__ == '__main__':
    test_poly()


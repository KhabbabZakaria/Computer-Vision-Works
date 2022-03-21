import cv2
import numpy as np
from typing import Tuple


def compute_harris_response(I: np.array, k: float = 0.06) -> Tuple[np.array]:

    Idx = cv2.Sobel(I,cv2.CV_32F,1,0)
    Idy = cv2.Sobel(I,cv2.CV_32F,0,1)

    Ixx = np.square(Idx)
    Iyy = np.square(Idy)
    Ixy = np.multiply(Idx, Idy)

    kernel_size = (3,3)
    std_dev = 1
    A = cv2.GaussianBlur(Ixx,kernel_size,std_dev)
    B = cv2.GaussianBlur(Iyy,kernel_size,std_dev)
    C = cv2.GaussianBlur(Ixy,kernel_size,std_dev)


    det = np.multiply(A, B) - np.square(C)
    Trace = A + B
    R = det - k* np.square(Trace)

    return (R, A, B, C, Idx, Idy)


def detect_corners(R: np.array, threshold: float = 0.1) -> Tuple[np.array, np.array]:

    print(R.shape)
    R = cv2.copyMakeBorder(R, 1, 1, 1, 1, cv2.BORDER_CONSTANT)


    # Step 2 (recommended) : create one image for every offset in the 3x3 neighborhood (6 lines).
    r,c = R.shape
    if r%3 == 1:
        r1 = 2
    elif r%3 == 2:
        r1 = 1
    else:
        r1 = 0
    if c%3 == 1:
        c1 = 2
    elif c%3 == 2:
        c1 = 1
    else:
        c1 = 0
    R = cv2.copyMakeBorder(R, 2*r1, 2*r1, 2*c1, 2*c1, cv2.BORDER_CONSTANT)
    r,c = R.shape
    image = R.reshape(r//3,3,c//3,3).swapaxes(1, 2).reshape(-1, 3, 3)

    max_vals = np.array([max(j) for j in [max(i) for i in image.tolist()]])


    single_where = np.where(([(image[i]>=max_vals[i]) & (image[i]> threshold) for i in range(image.shape[0])]),1,0)


    single_where = single_where.reshape(r//3,c//3, -3, 3)
    single_where = single_where.swapaxes(2, 1)
    single_where = single_where.reshape(r,c)

    nonzeros = np.transpose(np.nonzero(single_where))


    x = nonzeros[:,0]
    y = nonzeros[:,1]
    return (y,x)


def detect_edges(R: np.array, edge_threshold: float = -0.01) -> np.array:

    R = cv2.copyMakeBorder(R, 1, 1, 1, 1, cv2.BORDER_CONSTANT)


    mylist = []
    for j in range(R.shape[0]-2):
        for i in range(R.shape[1]-2):
            mylist.append(min(R[j+1][i], R[j+1][i+2]))

    x = np.array(mylist).reshape(R.shape[0]-2,R.shape[1]-2)
    mylist = []
    for j in range(R.shape[1]-2):
        for i in range(R.shape[0]-2):
            mylist.append(min(R[:,j+1][i], R[:,j+1][i+2]))

    y = np.array(mylist).reshape(R.shape[1]-2,R.shape[0]-2).T


    R = R[1:-1,1:-1]
    result = np.where(((R<x) | (R<y)) & (R<edge_threshold), True, False)


    return result

import numpy
import cv2
from typing import Tuple, Dict, List
import numpy as np
import scipy.spatial
from itertools import product

# These are typehints, they mostly make the code readable and testable
t_points = np.array
t_descriptors = np.array
t_homography = np.array
t_img = np.array
t_images = Dict[str, t_img]
t_homographies = Dict[Tuple[str, str], t_homography]  # The keys are the keys of src and destination images

np.set_printoptions(edgeitems=30, linewidth=180,
                    formatter=dict(float=lambda x: "%8.05f" % x))


def extract_features(img: t_img, num_features: int = 500) -> Tuple[t_points, t_descriptors]:

    orb = cv2.ORB_create(num_features, scoreType=cv2.ORB_FAST_SCORE)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    #print(keypoints)
    print(np.array(keypoints).shape)
    #print((keypts.pt[0], keypts.pt[1]) for keypts in keypoints]))
    #print(np.array([(keypts.pt[0], keypts.pt[1]) for keypts in keypoints]), np.array(descriptors))
    return np.array([(keypts.pt[0], keypts.pt[1]) for keypts in keypoints]), np.array(descriptors)


def filter_and_align_descriptors(f1: Tuple[t_points, t_descriptors], f2: Tuple[t_points, t_descriptors],
                                 similarity_threshold=.7, similarity_metric='hamming') -> Tuple[t_points, t_points]:

    dist_mat = scipy.spatial.distance.cdist(f1[1], f2[1], similarity_metric)

    indices = list(np.argmin(dist_mat, 0))


    mylist = []
    for i in range(len(indices)):
        mylist.append([indices[i], i])

    myarray = np.array(mylist)
    #print(myarray)

    dist_mat2 = np.sort(dist_mat, 0)

    srclist = []
    dstlist = []
    for i in range(dist_mat.shape[1]):
        d0 = dist_mat2[0][i] #minimum distance
        d1 = dist_mat2[1][i] #second minimum distance

        if d0/d1 <= similarity_threshold:
            srclist.append(myarray[i, 0])
            dstlist.append(myarray[i, 1])

    return (f1[0][srclist, :], f2[0][dstlist, :])


def compute_homography(f1: np.array, f2: np.array) -> np.array:

    homography_matrix = np.zeros((3, 3))

    A = np.zeros((2*len(f1), 9))
    for i in range(len(f1)):
        A[2*i] = [-f1[i][0], -f1[i][1], -1, 0, 0, 0, f1[i][0]*f2[i][0], f1[i][1]*f2[i][0], f2[i][0]]
        A[2*i+1] = [0, 0, 0, -f1[i][0], -f1[i][1], -1, f1[i][0]*f2[i][1], f1[i][1]*f2[i][1], f2[i][1]]

    _, _, V = np.linalg.svd(A)
    homography_matrix = V[8].reshape((3, 3))/V[8][8]

    return homography_matrix


def _get_inlier_count(src_points: np.array, dst_points: np.array, homography: np.array,
                      distance_threshold: float) -> int:

    inlier = 0

    for i in range(src_points.shape[0]):
        src_points2 = np.append(src_points[i], [1])

        projected = np.dot(homography, src_points2)

        projected = (projected / projected[2])[0:-1]

        norm = cv2.norm(dst_points[i], projected)

        if norm < distance_threshold:
            inlier = inlier + 1

    return inlier




def ransac(src_features: Tuple[t_points, t_descriptors], dst_features: Tuple[t_points, t_descriptors], steps,
           distance_threshold, n_points=4, similarity_threshold=.7) -> np.array:
    filter_and_align = filter_and_align_descriptors(src_features, dst_features, similarity_threshold)
    print(filter_and_align[0].shape)

    best_count = 0
    best_homography = np.eye(3)

    for n in range(steps):
        if n == steps - 1:
            print(f"Step: {n:4}  {best_count} RANSAC points match!")

        random_points = np.random.randint(0, filter_and_align[0].shape[0], size=n_points)

        random1 = []
        random2 = []

        for i in range(n_points):
            blah = np.random.randint(0, len(filter_and_align[0]))
            random1.append(filter_and_align[0][blah])
            random2.append(filter_and_align[1][blah])

        random1 = np.array(random1)
        random2 = np.array(random2)

        homos = compute_homography(random1, random2)

        inlier = _get_inlier_count(filter_and_align[0], filter_and_align[1], homos, distance_threshold)

        if inlier > best_count:
            best_count = inlier
            best_homography = homos

    return best_homography


def probagate_homographies(homographies: t_homographies, reference_name: str) -> t_homographies:

    initial = {k: v for k, v in homographies.items()}  # deep copy
    for k, h in list(initial.items()):
        initial[(k[1], k[0])] = np.linalg.inv(h)
    initial[(reference_name, reference_name)] = np.eye(3)  # Added the identity homography for the reference
    desired = set([(k[0], reference_name) for k in homographies.keys()])
    solved = {k: v for k, v in initial.items() if k[1] == reference_name}
    while not (set(solved.keys()) >= desired):

        new_steps = set([(i, s) for i, s in product(initial.keys(), solved.keys()) if
                     s[1] != i[0] and s[0] == i[1] and s[0] != s[1] and (i[0], s[1]) not in solved.keys()])


        assert len(new_steps) > 0  # not all desired can be linked to reference
        for initial_k, solved_k in new_steps:
            new_key = initial_k[0], solved_k[1]
            solved[solved_k]
            initial[initial_k]
            solved[new_key] = np.matmul(solved[solved_k], initial[initial_k])
    return solved


def compute_panorama_borders(images: t_images, homographies: t_homographies) -> Tuple[float, float, float, float]:

    homographies = {k[0]: v for k, v in homographies.items()}  # assining homographies to their source image
    assert homographies.keys() == images.keys()  # map homographies to source image only
    all_corners = []
    for name in sorted(images.keys()):
        img, homography = images[name], homographies[name]
        width, height = img.shape[0], img.shape[1]
        corners = ((0, 0), (0, width), (height, width), (height, 0))
        corners = np.array(corners, dtype='float32')
        all_corners.append(cv2.perspectiveTransform(corners[None, :, :], homography)[0, :, :])
    all_corners = np.concatenate(all_corners, axis=0)
    left, right = np.floor(all_corners[:, 0].min()), np.ceil(all_corners[:, 0].max())
    top, bottom = np.floor(all_corners[:, 1].min()), np.ceil(all_corners[:, 1].max())
    return left, top, right, bottom


def translate_homographies(homographies: t_homographies, dx: float, dy: float):

    trans_mat = np.array([[1,0,dx], [0,1,dy], [0,0,1]])

    homos2 = {}
    for key in homographies:
        homos2[key] = np.dot(trans_mat, homographies[key])

    return homos2



def stitch_panorama(images: t_images, homographies: t_homographies, output_size: Tuple[int, int],
                   rendering_order: List[str] = []) -> t_images:

    homographies = {k[0]: v for k, v in homographies.items()}  # assining homographies to their source image
    assert homographies.keys() == images.keys()
    if rendering_order == []:
        rendering_order = sorted(images.keys())
    panorama = np.zeros([output_size[1], output_size[0], 3], dtype=np.uint8)
    for name in rendering_order:
        rgba_img = cv2.cvtColor(images[name], cv2.COLOR_RGB2RGBA)
        rgba_img[:, :, 3] = 255
        tmp = cv2.warpPerspective(rgba_img, homographies[name], output_size, cv2.INTER_LINEAR_EXACT)
        new_pixels = ((tmp[:, :, 3] == 255)[:, :, None] & (panorama == np.zeros([1, 1, 3])))
        old_pixels = 1 - new_pixels
        panorama[:, :, :] = panorama * old_pixels + tmp[:, :, :3] * new_pixels
    return panorama


def create_stitched_image(images: t_images, homographies: t_homographies, reference_name: str,
                          rendering_order: List[str] = []):

    #  from homographies between consecutive images we compute all homographies from any image to the reference.
    homographies = probagate_homographies(homographies, reference_name=reference_name)
    #  lets calculate the panorama size
    left, top, right, bottom = compute_panorama_borders(images, homographies)
    width = int(1 + np.ceil(right) - np.floor(left))
    height = int(1 + np.ceil(bottom) - np.floor(top))
    #  lets make the homographies translate all images inside the panorama.
    homographies = translate_homographies(homographies, -left, -top)
    return stitch_panorama(images, homographies, (width, height), rendering_order=rendering_order)




"""
python implementation of the semi-global matching algorithm from Stereo Processing by Semi-Global Matching
and Mutual Information (https://core.ac.uk/download/pdf/11134866.pdf) by Heiko Hirschmuller.

author: David-Alexandre Beaupre
date: 2019/07/12
"""
import tensorflow as tf

import argparse
import sys
import time as t

import cv2
import numpy as np

from tensorflow.keras.models import Model, Sequential, load_model

from utils import normalize
import matchingCostModel
import aanetModel


class Direction:
    def __init__(self, direction=(0, 0), name='invalid'):
        """
        represent a cardinal direction in image coordinates (top left = (0, 0) and bottom right = (1, 1)).
        :param direction: (x, y) for cardinal direction.
        :param name: common name of said direction.
        """
        self.direction = direction
        self.name = name


# 8 defined directions for sgm
# These define the directions we calcualte costs for in a certain pixel
N = Direction(direction=(0, -1), name='north')
NE = Direction(direction=(1, -1), name='north-east')
E = Direction(direction=(1, 0), name='east')
SE = Direction(direction=(1, 1), name='south-east')
S = Direction(direction=(0, 1), name='south')
SW = Direction(direction=(-1, 1), name='south-west')
W = Direction(direction=(-1, 0), name='west')
NW = Direction(direction=(-1, -1), name='north-west')


class Paths:
    def __init__(self, mode=4):
        """
        represent the relation between the directions.
        """
        assert mode==4 or mode==8

        if mode == 4:
            self.paths = [N, E, S, W]
            self.size = len(self.paths)
            self.effective_paths = [(E,  W), (S, N)]
        
        elif mode == 8:
            self.paths = [N, NE, E, SE, S, SW, W, NW]
            self.size = len(self.paths)
            self.effective_paths = [(E,  W), (SE, NW), (S, N), (SW, NE)]


class Parameters:
    def __init__(self, max_disparity=64, P1=5, P2=70, csize=(7, 7), bsize=(3, 3)):
        """
        represent all parameters used in the sgm algorithm.
        :param max_disparity: maximum distance between the same pixel in both images.
        :param P1: penalty for disparity difference = 1
        :param P2: penalty for disparity difference > 1
        :param csize: size of the kernel for the census transform.
        :param bsize: size of the kernel for blurring the images and median filtering.
        """
        self.max_disparity = max_disparity
        self.P1 = P1
        self.P2 = P2
        self.csize = csize
        self.bsize = bsize


def load_images(left_name, right_name, parameters):
    """
    read and blur stereo image pair.
    :param left_name: name of the left image.
    :param right_name: name of the right image.
    :param parameters: structure containing parameters of the algorithm.
    :return: blurred left and right images.
    """

    # Note that Gaussian Blur is applied since it sharp details don't help much with determining depth... I guess xD

    # left = cv2.imread(left_name, 0)
    left = cv2.imread(left_name)
    left = cv2.GaussianBlur(left, parameters.bsize, 0, 0)
    # right = cv2.imread(right_name, 0)
    right = cv2.imread(right_name)
    right = cv2.GaussianBlur(right, parameters.bsize, 0, 0)
    return left, right


def get_indices(offset, dim, direction, height):
    """
    for the diagonal directions (SE, SW, NW, NE), return the array of indices for the current slice.
    :param offset: difference with the main diagonal of the cost volume.
    :param dim: number of elements along the path.
    :param direction: current aggregation direction.
    :param height: H of the cost volume.
    :return: arrays for the y (H dimension) and x (W dimension) indices.
    """
    y_indices = []
    x_indices = []

    for i in range(0, dim):
        if direction == SE.direction:
            if offset < 0:
                y_indices.append(-offset + i)
                x_indices.append(0 + i)
            else:
                y_indices.append(0 + i)
                x_indices.append(offset + i)

        if direction == SW.direction:
            if offset < 0:
                y_indices.append(height + offset - i)
                x_indices.append(0 + i)
            else:
                y_indices.append(height - i)
                x_indices.append(offset + i)

    return np.array(y_indices), np.array(x_indices)


def get_path_cost(slice, offset, parameters):
    """
    part of the aggregation step, finds the minimum costs in a D x M slice (where M = the number of pixels in the
    given direction)
    :param slice: M x D array from the cost volume.
    :param offset: ignore the pixels on the border.
    :param parameters: structure containing parameters of the algorithm.
    :return: M x D array of the minimum costs for a given slice in a given direction.
    """
    other_dim = slice.shape[0]
    disparity_dim = slice.shape[1]

    disparities = [d for d in range(disparity_dim)] * disparity_dim
    disparities = np.array(disparities).reshape(disparity_dim, disparity_dim)

    penalties = np.zeros(shape=(disparity_dim, disparity_dim), dtype=slice.dtype)
    penalties[np.abs(disparities - disparities.T) == 1] = parameters.P1
    penalties[np.abs(disparities - disparities.T) > 1] = parameters.P2

    minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=slice.dtype)
    minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

    for i in range(offset, other_dim):
        previous_cost = minimum_cost_path[i - 1, :]
        current_cost = slice[i, :]
        costs = np.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
        costs = np.amin(costs + penalties, axis=0)
        minimum_cost_path[i, :] = current_cost + costs - np.amin(previous_cost)
    return minimum_cost_path


def aggregate_costs(cost_volume, parameters, paths):
    """
    second step of the sgm algorithm, aggregates matching costs for N possible directions (4 in this case).
    :param cost_volume: array containing the matching costs.
    :param parameters: structure containing parameters of the algorithm.
    :param paths: structure containing all directions in which to aggregate costs.
    :return: H x W x D x N array of matching cost for all defined directions.
    """
    height = cost_volume.shape[0]
    width = cost_volume.shape[1]
    disparities = cost_volume.shape[2]
    start = -(height - 1)
    end = width - 1

    aggregation_volume = np.zeros(shape=(height, width, disparities, paths.size), dtype=cost_volume.dtype)

    path_id = 0
    for path in paths.effective_paths:
        # print('\tProcessing paths {} and {}...'.format(path[0].name, path[1].name), end='')
        print('\tProcessing paths {} and {}...'.format(path[0].name, path[1].name))
        sys.stdout.flush()
        dawn = t.time()

        main_aggregation = np.zeros(shape=(height, width, disparities), dtype=cost_volume.dtype)
        opposite_aggregation = np.copy(main_aggregation)

        main = path[0]
        if main.direction == S.direction:
            for x in range(0, width):
                south = cost_volume[0:height, x, :]
                north = np.flip(south, axis=0)
                main_aggregation[:, x, :] = get_path_cost(south, 1, parameters)
                opposite_aggregation[:, x, :] = np.flip(get_path_cost(north, 1, parameters), axis=0)

        if main.direction == E.direction:
            for y in range(0, height):
                east = cost_volume[y, 0:width, :]
                west = np.flip(east, axis=0)
                main_aggregation[y, :, :] = get_path_cost(east, 1, parameters)
                opposite_aggregation[y, :, :] = np.flip(get_path_cost(west, 1, parameters), axis=0)

        if main.direction == SE.direction:
            for offset in range(start, end):
                south_east = cost_volume.diagonal(offset=offset).T
                north_west = np.flip(south_east, axis=0)
                dim = south_east.shape[0]
                y_se_idx, x_se_idx = get_indices(offset, dim, SE.direction, None)
                y_nw_idx = np.flip(y_se_idx, axis=0)
                x_nw_idx = np.flip(x_se_idx, axis=0)
                main_aggregation[y_se_idx, x_se_idx, :] = get_path_cost(south_east, 1, parameters)
                opposite_aggregation[y_nw_idx, x_nw_idx, :] = get_path_cost(north_west, 1, parameters)

        if main.direction == SW.direction:
            for offset in range(start, end):
                south_west = np.flipud(cost_volume).diagonal(offset=offset).T
                north_east = np.flip(south_west, axis=0)
                dim = south_west.shape[0]
                y_sw_idx, x_sw_idx = get_indices(offset, dim, SW.direction, height - 1)
                y_ne_idx = np.flip(y_sw_idx, axis=0)
                x_ne_idx = np.flip(x_sw_idx, axis=0)
                main_aggregation[y_sw_idx, x_sw_idx, :] = get_path_cost(south_west, 1, parameters)
                opposite_aggregation[y_ne_idx, x_ne_idx, :] = get_path_cost(north_east, 1, parameters)

        aggregation_volume[:, :, :, path_id] = main_aggregation
        aggregation_volume[:, :, :, path_id + 1] = opposite_aggregation
        path_id = path_id + 2

        dusk = t.time()
        print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return aggregation_volume


# This is the Matching Cost Computation and this one uses Census Transform and Hamming Dist
# TODO: This is where we would insert our awesome net
def compute_costs(left, right, parameters, save_images, model):
    """
    first step of the sgm algorithm, matching cost based on census transform and hamming distance.
    :param left: left image.
    :param right: right image.
    :param parameters: structure containing parameters of the algorithm.
    :param save_images: whether to save census images or not.
    :return: H x W x D array with the matching costs.
    """
    assert left.shape[0] == right.shape[0] and left.shape[1] == right.shape[1], 'left & right must have the same shape.'
    assert parameters.max_disparity > 0, 'maximum disparity must be greater than 0.'

    # height = left.shape[0]
    # width = left.shape[1]
    # cheight = parameters.csize[0]
    # cwidth = parameters.csize[1]
    # y_offset = int(cheight / 2)
    # x_offset = int(cwidth / 2)
    # disparity = parameters.max_disparity

    # left_img_census = np.zeros(shape=(height, width), dtype=np.uint8)
    # right_img_census = np.zeros(shape=(height, width), dtype=np.uint8)
    # left_census_values = np.zeros(shape=(height, width), dtype=np.uint64)
    # right_census_values = np.zeros(shape=(height, width), dtype=np.uint64)

    # print('\tComputing left and right census...')
    # sys.stdout.flush()
    # dawn = t.time()
    # # pixels on the border will have no census values
    # for y in range(y_offset, height - y_offset):
    #     for x in range(x_offset, width - x_offset):
    #         left_census = np.int64(0)
    #         center_pixel = left[y, x]
    #         reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int64)
    #         image = left[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
    #         comparison = image - reference
    #         for j in range(comparison.shape[0]):
    #             for i in range(comparison.shape[1]):
    #                 if (i, j) != (y_offset, x_offset):
    #                     left_census = left_census << 1
    #                     if comparison[j, i] < 0:
    #                         bit = 1
    #                     else:
    #                         bit = 0
    #                     left_census = left_census | bit
    #         left_img_census[y, x] = np.uint8(left_census)
    #         left_census_values[y, x] = left_census

    #         right_census = np.int64(0)
    #         center_pixel = right[y, x]
    #         reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int64)
    #         image = right[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
    #         comparison = image - reference
    #         for j in range(comparison.shape[0]):
    #             for i in range(comparison.shape[1]):
    #                 if (i, j) != (y_offset, x_offset):
    #                     right_census = right_census << 1
    #                     if comparison[j, i] < 0:
    #                         bit = 1
    #                     else:
    #                         bit = 0
    #                     right_census = right_census | bit
    #         right_img_census[y, x] = np.uint8(right_census)
    #         right_census_values[y, x] = right_census

    # dusk = t.time()
    # print('\t(done in {:.2f}s)'.format(dusk - dawn))

    # if save_images:
    #     cv2.imwrite('left_census.png', left_img_census)
    #     cv2.imwrite('right_census.png', right_img_census)

    # # print('\tComputing cost volumes...', end='')
    # print('\tComputing cost volumes...')
    # sys.stdout.flush()
    # dawn = t.time()
    # left_cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
    # right_cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
    # lcensus = np.zeros(shape=(height, width), dtype=np.int64)
    # rcensus = np.zeros(shape=(height, width), dtype=np.int64)

    # # Calculate disparity from 1 to Dmax
    # for d in range(0, disparity):
    #     rcensus[:, (x_offset + d):(width - x_offset)] = right_census_values[:, x_offset:(width - d - x_offset)]
    #     left_xor = np.int64(np.bitwise_xor(np.int64(left_census_values), rcensus))
    #     left_distance = np.zeros(shape=(height, width), dtype=np.uint32)
    #     while not np.all(left_xor == 0):
    #         tmp = left_xor - 1
    #         mask = left_xor != 0
    #         left_xor[mask] = np.bitwise_and(left_xor[mask], tmp[mask])
    #         left_distance[mask] = left_distance[mask] + 1
    #     left_cost_volume[:, :, d] = left_distance

    #     # Right image cost computation
    #     # lcensus[:, x_offset:(width - d - x_offset)] = left_census_values[:, (x_offset + d):(width - x_offset)]
    #     # right_xor = np.int64(np.bitwise_xor(np.int64(right_census_values), lcensus))
    #     # right_distance = np.zeros(shape=(height, width), dtype=np.uint32)
    #     # while not np.all(right_xor == 0):
    #     #     tmp = right_xor - 1
    #     #     mask = right_xor != 0
    #     #     right_xor[mask] = np.bitwise_and(right_xor[mask], tmp[mask])
    #     #     right_distance[mask] = right_distance[mask] + 1
    #     # right_cost_volume[:, :, d] = right_distance

    # dusk = t.time()
    # print('\t(done in {:.2f}s)'.format(dusk - dawn))

    # return left_cost_volume, right_cost_volume

    # left_cost_volume = model.predict([np.asarray([np.swapaxes(left_image, 0, 1)]), np.asarray([np.swapaxes(right_image, 0, 1)])])
    

    if model == "net1":
        print('\tRunning Matching Cost Model')
        sys.stdout.flush()
        dawn = t.time()
        left_cost_volume = matchingCostModel.getCostVolume("matchingCostModel.h5", left, right)
        dusk = t.time()
        print('\t(done in {:.2f}s)'.format(dusk - dawn))

    if model == "aanet":
        print('\tRunning AANet Model')
        sys.stdout.flush()
        dawn = t.time()
        left_cost_volume = aanetModel.getCostVolume("./aanet/pretrained/aanet+_kitti15-2075aea1.pth", left, right)
        dusk = t.time()
        print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return left_cost_volume


# TODO: This is where we would also return a 2nd disparity map using sedong argmin
def select_disparity(aggregation_volume):
    """
    last step of the sgm algorithm, corresponding to equation 14 followed by winner-takes-all approach.
    :param aggregation_volume: H x W x D x N array of matching cost for all defined directions.
    :return: disparity image.
    """
    volume = np.sum(aggregation_volume, axis=3)
    # disparity_map = np.argmin(volume, axis=2)

    # Sort volume matrix
    volume = np.argsort(volume, axis=2)

    disparity_map = volume[:,:,0]
    sub_obtimal_disparity_map = volume[:,:,1]

    return disparity_map, sub_obtimal_disparity_map


def normalize(volume, parameters):
    """
    transforms values from the range (0, 64) to (0, 255).
    :param volume: n dimension array to normalize.
    :param parameters: structure containing parameters of the algorithm.
    :return: normalized array.
    """
    return 255.0 * volume / parameters.max_disparity


def get_recall(disparity, gt, args):
    """
    computes the recall of the disparity map.
    :param disparity: disparity image.
    :param gt: path to ground-truth image.
    :param args: program arguments.
    :return: rate of correct predictions.
    """
    truth = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
    truth = cv2.resize(truth, (187, 621), interpolation = cv2.INTER_AREA)
    # gt = np.float32(np.swapaxes(truth, 0, 1))
    gt = np.int16(gt / 255.0 * float(args.disp))
    disparity = np.int16(np.float32(disparity) / 255.0 * float(args.disp))
    correct = np.count_nonzero(np.abs(disparity - gt) <= 3)
    return float(correct) / gt.size


def sgm():
    """
    main function applying the semi-global matching algorithm.
    :return: void.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', default='test/left/0001.png', help='name (path) to the left image')
    parser.add_argument('--right', default='test/right/0001.png', help='name (path) to the right image')
    parser.add_argument('--left_gt', default='test/prediction/0001.png', help='name (path) to the left ground-truth image')
    parser.add_argument('--right_gt', default='test/prediction/0001.png', help='name (path) to the right ground-truth image')
    parser.add_argument('--D1', default='output/disparity_map_MODEL.png', help='name (path) of the output disparity map')
    parser.add_argument('--D2', default='output/sub_obtimal_disparity_map_MODEL.png', help='name (path) of the output sub optimal disparity map')
    parser.add_argument('--disp', default=64, type=int, help='maximum disparity for the stereo pair')
    parser.add_argument('--paths', default=4, type=int, help='number of paths for cost aggregation')
    parser.add_argument('--images', default=False, type=bool, help='save intermediate representations')
    parser.add_argument('--eval', default=True, type=bool, help='evaluate disparity map with 3 pixel error')
    parser.add_argument('--model', default="aanet", type=str, help='evaluate disparity map with 3 pixel error')
    args = parser.parse_args()

    left_name = args.left
    right_name = args.right

    # Used for evaluation only
    left_gt_name = args.left_gt
    right_gt_name = args.right_gt

    D1 = args.D1
    D2 = args.D2
    disparity = args.disp
    paths = args.paths
    save_images = args.images
    evaluation = args.eval

    model = args.model

    if model not in ["aanet", "net1"]:
        raise("Model does not exist")


    dawn = t.time()

    parameters = Parameters(max_disparity=disparity, P1=2.3, P2=55.8, csize=(7, 7), bsize=(3, 3))
    paths = Paths(paths)

    if model == "net1":
        # Load images
        print('\nLoading images...')
        left, right = load_images(left_name, right_name, parameters)

        # Calculate cost using network
        print('\nStarting cost computation...')
        left_cost_volume = compute_costs(left, right, parameters, save_images, model)

        if save_images:
            left_disparity_map = np.uint8(normalize(np.argmin(left_cost_volume, axis=2), parameters))
            cv2.imwrite('output/disp_map_left_cost_volume.png', left_disparity_map)
            right_disparity_map = np.uint8(normalize(np.argmin(right_cost_volume, axis=2), parameters))
            cv2.imwrite('output/disp_map_right_cost_volume.png', right_disparity_map)

        # Do SGM cost aggregation on left volume matrix
        print('\nStarting left aggregation computation...')
        left_aggregation_volume = aggregate_costs(left_cost_volume, parameters, paths)

        # Calculate WTA and sub optimal disparity maps
        print('\nSelecting best disparities...')
        disparity_map, sub_obtimal_disparity_map = select_disparity(left_aggregation_volume)

        # Do standard disparity map normalization
        disparity_map = np.uint8(normalize(disparity_map, parameters))
        sub_obtimal_disparity_map = np.uint8(normalize(sub_obtimal_disparity_map, parameters))

        if save_images:
            cv2.imwrite('output/left_disp_map_no_post_processing.png', left_disparity_map)
            cv2.imwrite('output/right_disp_map_no_post_processing.png', right_disparity_map)

        # Apply median filter to both maps
        print('\nApplying median filter...')
        disparity_map = cv2.medianBlur(disparity_map, parameters.bsize[0])
        sub_obtimal_disparity_map = cv2.medianBlur(sub_obtimal_disparity_map, parameters.bsize[0])

        # Save disparity maps
        cv2.imwrite(D1, disparity_map)
        cv2.imwrite(D2, sub_obtimal_disparity_map)


        # Evaluate disparity maps to ground truths
        if evaluation:
            print('\nEvaluating optimal disparity map...')
            recall = get_recall(disparity_map, left_gt_name, args)
            print('\tRecall = {:.2f}%'.format(recall * 100.0))
            print('\nEvaluating sub obtimal disparity map...')
            recall = get_recall(sub_obtimal_disparity_map, left_gt_name, args)
            print('\tRecall = {:.2f}%'.format(recall * 100.0))


    if model == "aanet":
        # Load images
        print('\nLoading images...')
        left, right = load_images(left_name, right_name, parameters)

        # Calculate Disparity Map
        print('\nRunnin AANet Model...')
        aanet_dawn = t.time()
        optimal_disparity_map = aanetModel.getDisparityMap("./aanet/pretrained/aanet+_kitti15-2075aea1.pth", left, right)
        aanet_dusk = t.time()
        print('\t(done in {:.2f}s)'.format(aanet_dusk - aanet_dawn))

        cv2.imwrite(D1, optimal_disparity_map)
        
    
    dusk = t.time()
    print('\nEnd.')
    print('\nTotal execution time = {:.2f}s'.format(dusk - dawn))


if __name__ == '__main__':
    sgm()

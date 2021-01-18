
import torch
import cv2
import numpy as np
# from utils import read_txt
import os

def read_img_reverse(img, coords, types, num_types=5, grid_num=10) -> torch.Tensor:
    '''
    Convert image with bbox predictions as into grid format
    :param img: image path in str or image in np.ndarray
    :param coords: Nx4 tensor/np.ndarray for box coords
    :param types: Nx1 tensor/np.ndarray for box types (logo, input etc.)
    :param num_types: total number of box types
    :param grid_num: number of grids needed
    :return: grid tensor
    '''
    
    img = cv2.imread(img) if not isinstance(img, np.ndarray) else img
    coords = coords.numpy() if not isinstance(coords, np.ndarray) else coords
    types = types.numpy() if not isinstance(types, np.ndarray) else types
    
    # Incorrect path/empty image
    if img is None:
        raise AttributeError('Image is None')
        
    height, width = img.shape[:2]
    
    # Empty image
    if height == 0 or width == 0:
        raise AttributeError('Empty image')

    # grid array of shape CxHxW
    grid_arrs = np.zeros((4+num_types, grid_num, grid_num))  # Must be [0, 1], use rel_x, rel_y, rel_w, rel_h

    for j, coord in enumerate(coords):
        x1, y1, x2, y2 = coord
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        if w == 0 or h == 0:
            continue # ignore

        # get the assigned grid index
        assigned_grid_w, assigned_grid_h = int(((x1 + x2)/2) // (width // grid_num)), int(((y1 + y2)/2) // (height // grid_num))

        # if this grid has been assigned before, check whether need to re-assign
        if grid_arrs[0, assigned_grid_h, assigned_grid_w] != 0: # visted
            exist_type = np.where(grid_arrs[:, assigned_grid_h, assigned_grid_w] == 1)[0][0] - 4
            new_type = types[j]
            if new_type > exist_type: # if new type has lower priority than existing type
                continue

        # fill in rel_xywh
        grid_arrs[0, assigned_grid_h, assigned_grid_w] = float(x1/width)
        grid_arrs[1, assigned_grid_h, assigned_grid_w] = float(y1/height)
        grid_arrs[2, assigned_grid_h, assigned_grid_w] = float(w/width)
        grid_arrs[3, assigned_grid_h, assigned_grid_w] = float(h/height)

        # one-hot encoding for type
        cls_arr = np.zeros(num_types)
        cls_arr[types[j]] = 1

        grid_arrs[4:, assigned_grid_h, assigned_grid_w] = cls_arr

    return torch.from_numpy(grid_arrs)

def read_img(img_path, coords, types, num_types=5, grid_num=10):
    '''
    Convert image with bbox predictions as into grid format
    :param img: image path in str or image in np.ndarray
    :param coords: Nx4 tensor/np.ndarray for box coords
    :param types: Nx1 tensor/np.ndarray for box types (logo, input etc.)
    :param num_types: total number of box types
    :param grid_num: number of grids needed
    :return: grid tensor
    '''
    
    img = cv2.imread(img_path) if not isinstance(img_path, np.ndarray) else img_path
    coords = coords.numpy() if not isinstance(coords, np.ndarray) else coords
    types = types.numpy() if not isinstance(types, np.ndarray) else types

    if img is None:
        raise AttributeError('Image is None')
    height, width = img.shape[:2]
    if height == 0 or width == 0:
        raise AttributeError('Empty image')

    # grid array of shape CxHxW
    grid_arrs = np.zeros((4+num_types, grid_num, grid_num))  # Must be [0, 1], use rel_x, rel_y, rel_w, rel_h
    type_dict = {'logo': 1, 'input': 2, 'button': 3, 'label': 4, 'block':5}

    for j, coord in enumerate(coords):
        x1, y1, x2, y2 = coord
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        if w == 0 or h == 0:
            continue # incorrect coordinate

        # get the assigned grid index
        assigned_grid_w, assigned_grid_h = int(((x1 + x2)/2) // (width // grid_num)), int(((y1 + y2)/2) // (height // grid_num))

        # if this grid has been assigned before, check whether need to re-assign
        if grid_arrs[0, assigned_grid_h, assigned_grid_w] != 0: # visted
            exist_type = np.where(grid_arrs[:, assigned_grid_h, assigned_grid_w] == 1)[0][0] - 3
            new_type = type_dict[types[j]]
            if new_type > exist_type: # if new type has lower priority than existing type
                continue

        # fill in rel_xywh
        grid_arrs[0, assigned_grid_h, assigned_grid_w] = float(x1/width)
        grid_arrs[1, assigned_grid_h, assigned_grid_w] = float(y1/height)
        grid_arrs[2, assigned_grid_h, assigned_grid_w] = float(w/width)
        grid_arrs[3, assigned_grid_h, assigned_grid_w] = float(h/height)

        # one-hot encoding for type
        cls_arr = np.zeros(num_types)
        cls_arr[type_dict[types[j]] - 1] = 1

        grid_arrs[4:, assigned_grid_h, assigned_grid_w] = cls_arr

    return torch.from_numpy(grid_arrs)


if __name__ == '__main__':

    num_imgs, labels, paths, preprocess_coordinates, types = read_txt('./data/first_round_3k3k/all_coords.txt')
    check_path = paths[500]
    preprocess_coordinates = np.asarray(preprocess_coordinates)[np.asarray(paths) == check_path]
    types = np.asarray(types)[np.asarray(paths) == check_path]

    grid_arr = read_img(img_path=os.path.join('./data/first_round_3k3k/credential', check_path+'.png'),
              coords=preprocess_coordinates,
              types=types)

    print(grid_arr.shape)


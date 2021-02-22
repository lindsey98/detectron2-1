import torch.utils.data as data
import numpy as np
from bit_pytorch.grid_divider import read_img
from bit_pytorch.utils import read_txt, read_txt_screenshot
import torchvision.transforms as transform
import os
import torch

import matplotlib.pyplot as plt
from PIL import Image
import cv2

class GetLoader(data.Dataset):
    '''
    Define customized dataloader
    '''

    def __init__(self, img_folder: str, annot_path: str):
        self.img_folder = img_folder
        self.annot_path = annot_path
        self.num_imgs, self.labels, self.paths, self.preprocess_coordinates, self.img_classes = read_txt(annot_path)
        self.classes = {'credential': 0, 'noncredential': 1}

    def __getitem__(self, item: int):

        image_file = list(set(self.paths))[item] # image path
        img_coords = np.asarray(self.preprocess_coordinates)[np.asarray(self.paths) == image_file] # box coordinates
        img_classes = np.asarray(self.img_classes)[np.asarray(self.paths) == image_file] # box types
        if len(img_coords) == 0:
            raise IndexError('list index out of range')

        img_label = self.classes[np.asarray(self.labels)[np.asarray(self.paths) == image_file][0]] # credential/non-credential

        grid_arr = read_img(img_path=os.path.join(self.img_folder, image_file+'.png'),
                            coords=img_coords, types=img_classes, grid_num=10)

        return grid_arr, img_label

    def __len__(self):
        return self.num_imgs


class ImageLoader(data.Dataset):

    def __init__(self, img_folder: str, annot_path: str):
        self.img_folder = img_folder
        self.annot_path = annot_path
        self.num_imgs, self.labels, self.paths = read_txt_screenshot(annot_path)
        self.classes = {'credential': 0, 'noncredential': 1}
        self.transform = transform.Compose([transform.Resize((256, 256)),
                                            transform.ToTensor()])

    def __getitem__(self, item: int):

        image_file = list(set(self.paths))[item] # image path

        img_label = self.classes[np.asarray(self.labels)[np.asarray(self.paths) == image_file][0]] # credential/non-credential

        image = Image.open(os.path.join(self.img_folder, image_file+'.png')).convert('RGB')

        image = self.transform(image)

        return image, img_label

    def __len__(self):
        return self.num_imgs

if __name__ == '__main__':
    
    train_set_orig = ImageLoader(img_folder='../datasets/train_imgs',
                            annot_path='../datasets/train_coords.txt')
        
    train_set = ImageLoader(img_folder='../datasets/train_merge_imgs',
                            annot_path='../datasets/train_merge_coords.txt')

    val_set = ImageLoader(img_folder='../datasets/val_imgs',
                          annot_path='../datasets/val_coords.txt')
    
    print(len(train_set_orig))
    print(len(train_set))
    print(len(val_set))
#     train_loader = torch.utils.data.DataLoader(
#         train_set, batch_size=32, drop_last=False, shuffle=False)

#     val_loader = torch.utils.data.DataLoader(
#         test_set, batch_size=32, drop_last=False, shuffle=False)

    # from bit_pytorch.train import recycle
    # for x, y in recycle(train_loader):
    #     print(y)
    #
    # # print(len(train_set))
    # for x, y in train_loader:
    #     # print(x)
    #     print(x.shape)
    #     print(y)
    #     break
    #
    # for x, y in val_loader:
    #     # print(x)
    #     print(x.shape)
    #     print(y)
    #     break
    #
    #
    # #     plt.imshow(Image.open(os.path.join('./data/first_round_3k3k/all_imgs', file_path[0]+'.png')))
    # #     plt.show()
    # #     break
    # #     print(x)
    # #     print(y)
    # #     break

## utils for Face Segmentation
import io
import torch
import torch.nn as nn
import torch.nn.functional as Face
import numpy as np
import cv2
from PIL import Image

## Blend images with their corresponding segmentation prediction
def blend_seg_pred(img, seg, alpha=0.5):
    """
    ****INPUT****:
    img(torch.tensor): a batch of image tensors of shape (batch, 3, height, width)
    seg(torch.tensor): a batch of segmantation labels of shape (batch, height, width)
    alpha(float): opacity value for the segmentation in the range [0, 1]
                - 0: completely transparent
                - 1: completely opaque

    ****RETURN****:
    blend(torch.tensor): the blended image
    """
    pred = seg.argmax(1)
    pred = pred.view(pred.shape[0], 1, pred.shape[1], pred.shape[2]).repeat(1, 3, 1, 1)
    blend = img

    # for each segmentation class except the background (label 0)
    for i in range(1, seg.shape[1]):
        color_mask = -torch.ones_like(img)
        color_mask[:, -i, :, :] = 1
        alpha_mask = 1 - (pred == i).float() * alpha
        blend = blend * alpha_mask + color_mask * (1 - alpha_mask)

    return blend

## Blend images with their corresponding segmentation label
def blend_seg_label(img, seg, alpha=0.5):
    """
    ****INPUT****:
    img(torch.tensor): a batch of image tensors of shape (batch, 3, height, width)
    seg(torch.tensor): a batch of segmantation labels of shape (batch, height, width)
    alpha(float): opacity value for the segmentation in the range [0, 1]
                - 0: completely transparent
                - 1: completely opaque

    ****RETURN****:
    blend(torch.tensor): the blended image
    """
    pred = seg.unsqueeze(1).repeat(1, 3, 1, 1)
    blend = img

    # for each segmentation class except the background (label 0)
    for i in range(1, pred.shape[1]):
        color_mask = -torch.ones_like(img)
        color_mask[:, -i, :, :] = 1
        alpha_mask = 1 - (pred == i).float() * alpha
        blend = blend * alpha_mask + color_mask * (1 - alpha_mask)

    return blend

## Simulated random hair occlusions on face mask
def random_hair_inpainting_mask(face_mask):
    """
    ****ALGORITHM****:
    1. Randomly choose a 'y' coordinate of the face mask
    2. Randomly choose a 'x' coordinate (either minimum or maximum 'x' value of the selected line)
    3. A random ellipse is rendered with its center in (x, y)
    4. The inpainting map is the intersection of the face mask with the ellipse

    ****INPUT****:
    fask_mask(np.array): a binary mask tensor of shape (H, W)
                        - 1: face region
                        - 0: background
    
    ****RETURN****:
    inpainting_mask(np.array): result mask
    """
    mask = face_mask == 1
    inpainting_mask = np.zeros(mask.shape, 'uint8')
    a = np.where(mask != 0)

    if len(a[0]) == 0 or len(a[1]) == 0:
        return inpainting_mask
    if (np.max(a[0]) - np.min(a[0])) <= 10 or (np.max(a[1]) - np.min(a[1])) <= 10:
        return inpainting_mask

    # select a random point on the mask borders
    try:
        y_coords = np.unique(a[0])
        y_ind = np.random.randint(len(y_coords))
        y = y_coords[y_ind]
        x_ind = np.where(a[0] == y)
        x_coords = a[1][x_ind[0]]
        x = x.x_coords.min() if np.random.rand() > 0.5 else x_coords.max()
    except:
        print(y_coords)
        print(x_coords)

    # draw inpainting shape
    width = a[1].max() - a[1].min() + 1
    #hegith = a[0].max() - a[0].min() + 1
    scale = (np.random.randint(width // 4, width // 2), np.random.randint(width // 4, width // 2))
    rotation_angle = np.random.randint(0, 180)
    cv2.ellipse(inpainting_mask, (x, y), scale, rotation_angle, 0, 360, (255, 255, 255), -1, 8)

    # get inpainting mask by intersection with face mask
    inpainting_mask *= mask
    inpainting_mask = inpainting_mask > 0

    #cv2.imwrite('face_mask', inpainting_mask)
    return inpainting_mask
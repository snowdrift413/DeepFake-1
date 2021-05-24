## Face Landmarks utilities
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

## Horizontal Flip 68 points landmarks
def horizontal_flip_face_landmarks_68pts(landmarks, width):
    """
    ****INPUT****:
    landmarks(np.array): Landmarks points of shape (68, 2)
    width(int): The width of the corresponding image
    
    ****RETURN****:
    landmarks(np.array): Horizontally flipped landmarks
    """
    assert landmarks.shape[0] == 68
    landmarks = landmarks.copy()

    # Invert X coordinates
    for p in landmarks:
        p[0] = width - p[0]

    # Jaw
    right_jaw = list(range(0, 8))
    left_jaw = list(range(16, 8, -1))
    landmarks[right_jaw + left_jaw] = landmarks[left_jaw + right_jaw]

    # Eyebrows
    right_bow = list(range(17, 22))
    left_bow = list(range(26, 21, -1))
    landmarks[right_bow + left_bow] = landmarks[left_bow + right_bow]

    # Nose
    right_nostril = list(range(31, 33))
    left_nostril = list(range(35, 33, -1))
    landmarks[right_nostril + left_nostril] = landmarks[left_nostril + right_nostril]

    # Eyes
    right_eye = list(range(36, 42))
    left_eye = [45, 44, 43, 42, 47, 46]
    landmarks[right_eye + left_eye] = landmarks[left_eye + right_eye]

    # Mouth Outer
    mouth_out_right = [48, 49, 50, 59, 58]
    mouth_out_left = [54, 53, 52, 55, 56]
    landmarks[mouth_out_right + mouth_out_left] = landmarks[mouth_out_left + mouth_out_right]

    # Mouth Inner
    mouth_in_right = [60, 61, 67]
    mouth_in_left = [64, 63, 65]
    landmarks[mouth_in_right + mouth_in_left] = landmarks[mouth_in_left + mouth_in_right]

    return landmarks
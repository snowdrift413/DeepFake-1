import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
import torchvision.transforms.functional as F

def rgb2tensor(img, normalize=True):
    #if type(img) == list or tuple:
    if isinstance(img, (list, tuple)):
        return [rgb2tensor(i) for i in img]
    tensor = F.to_tensor(img)
    if normalize:
        tensor = F.normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    return tensor.unsqueeze(0)

def bgr2tensor(img, normalize=True):
    if isinstance(img, (list, tuple)):
        return [bgr2tensor(i, normalize) for i in img]
    
    return rgb2tensor(img[:, :, ::-1].copy(), normalize)

def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add(m)
    
    return tensor

def tensor2rgb(tensor):
    output = unnormalize(tensor.clone(), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    output = output.unsqueeze().permute(1, 2, 0).cpu().numpy()
    output = np.round(output * 255).astype('uint8')

    return output

def tensor2bgr(tensor):
    output = tensor2rgb(tensor)
    output = output[:, :, ::-1]

    return output

def make_grid(*args, cols=8):
    assert len(args) > 0, '적어도 하나 이상의 텐서를 인풋해줘야 합니다!, At least one input tensor must be given!'
    imgs = torch.cat([a.cpu() for a in args], dim=2)

    return torchvision.utils.make_grid(imgs, nrow=cols, normalize=True, scale_each=False)

def create_pyramid(img, n=1):
    if isinstance(img, (list, tuple)):
        return img
    
    pyramid = [img]
    for i in range(n - 1):
        pyramid.append(nn.functional.avg_pool2d(pyramid[-1], 3, stride=2, padding=1, count_include_pad=False))

    return pyramid
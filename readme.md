# DeepFake

## **1. FSGAN: Subject Agnostic Face Swapping and Reenactment**
paper link: https://arxiv.org/abs/1908.05932

## **1.1 Train**
```
& python main.py --
```

## **1.2 Test**
```
& python main.py --
```

## **Directors structure**
```
FACE-SWAPPING
+---[losses]
|   +---__init__.py
|   |---gan_loss.py
|   |---vgg_loss.py
|
+---[models]
|   +---__init__.py
|   |---unet.py
|
+---[utils]
|   +---__init__.py
|   |---bbox_utils.py
|   |---image_utils.py
```
import random
import PIL
from PIL import Image
import numpy as np
from PIL import ImageDraw

def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.05 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.05,1.9]
    assert 0.05 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.05,1.9]
    assert 0.05 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.05,1.9]
    assert 0.05 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def Identity(img, v):
    return img


def augment_list():  # 16 oeprations and their ranges
    # 这些增强的方式都是在图片上创建一个类对象，然后在修改图片的属性的参数值
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, 0, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, 0, 0.3),
        (ShearY, 0, 0.3),
        (Solarize, 0, 256),
        (TranslateX, 0, 0.3),
        (TranslateY, 0, 0.3)
    ]
    return l

def Cutout(image,val):
    assert 0 <= val <= 0.3
    if val == 0:
        return image
    val = val * image.size[0]
    return mask_image(image, val)

def mask_image(img,val):
    if val <= 0:
        return img
    w, h = img.size
    random_w = np.random.uniform(w)
    random_h = np.random.uniform(h)
    w0 = int(max(0, random_w - val / 2))
    h0 = int(max(0, random_h - val / 2))
    w1 = int(min(w, w0 + val))
    h1 = int(min(h, h0 + val))
    wh = (w0, h0, w1, h1)
    color = (125, 123, 114)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(wh,color)
    return img


if __name__ == '__main__':
    image_path = 'D:\\python\\pytorch作业\\all_data\\hotdog\\test\\hotdog\\1000.png'
    a = Image.open(image_path)
    b = Cutout(a,val=0.2)
    b.show()


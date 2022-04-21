import os
import os.path as osp
import cv2
import glob
import pdb
import random
import copy
import json
from tqdm import tqdm
import numpy as np
from albumentations import (
    CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur, PiecewiseAffine, ChannelShuffle, RGBShift, Cutout, InvertImg,
    Sharpen, Emboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# https://blog.csdn.net/qq_43474959/article/details/109849066
def imgBrightness(img1, c, b):
    rows, cols, channels = img1.shape
    blank = np.zeros([rows, cols, channels], img1.dtype)
    rst = cv2.addWeighted(img1, c, blank, 1-c, b)
    return rst

# augment character pic
def pic_aug(p=0.8):
    return Compose([
        RandomRotate90(),
        Flip(p),
        Transpose(),
        GaussNoise(p=0.5),
        OneOf([
            MotionBlur(p=0.7),
            MedianBlur(blur_limit=9, p=0.5),
            Blur(blur_limit=10, p=0.5),
        ], p=0.8),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, border_mode=cv2.BORDER_REPLICATE, p=0.8),
        OneOf([
            OpticalDistortion(p=0.6),
            # GridDistortion(p=0.6),
            PiecewiseAffine(p=0.6),
        ], p=0.8),
        OneOf([
            CLAHE(clip_limit=2),
            Sharpen(),
            Emboss(),
            RandomBrightnessContrast(),
        ], p=0.8),
        OneOf(
            [
                HueSaturationValue(p=0.8),
                ChannelShuffle(p=0.9),
                RGBShift(p=0.95)
            ], p=0.8
        )
    ], p=p)


# compute iou
def compute_iou(rec_mat, rec):
    rec = rec.repeat(rec_mat.shape[0], 0)
    area1 = (rec_mat[..., 2] - rec_mat[..., 0]) * (rec_mat[..., 3] - rec_mat[..., 1])
    area2 = (rec[..., 2] - rec[..., 0]) * (rec[..., 3] - rec[..., 1])
    lt = np.max(np.stack([rec_mat[..., 0:2], rec[..., 0:2]], axis=-1), axis=-1)
    rb = np.min(np.stack([rec_mat[..., 2:], rec[..., 2:]], axis=-1), axis=-1)
    _wh = rb - lt
    intersect = _wh[:, 0] * _wh[:, 1]
    iou = intersect / (area1 + area2 - intersect)
    if (iou > 0.3).any():
        return False
    else:
        return True


# generate character
def rndChar():
    return chr(random.randint(65, 69))


def rndColor(type):
    if type == 1:
        return random.randint(0, 125), random.randint(0, 125), random.randint(0, 125)
    elif type == 2:
        return random.randint(126, 254), random.randint(126, 254), random.randint(126, 254)


def get_character_img():
    # generate single character
    fontSize = 120
    width = fontSize
    height = fontSize
    # generate image
    image = Image.new('RGB', (int(width), int(height + 10)), (211, 211, 211))
    # generate font
    font_list = ['C:\Windows\Fonts\swromnc.ttf', 'C:\Windows\Fonts\swtxt.ttf',
                 'C:\Windows\Fonts\swcomp.ttf', 'C:/Windows/Fonts/timesi.ttf', 'C:\Windows\Fonts\msyi.ttf']
    # chosoe font
    font_file = random.choice(font_list)
    font = ImageFont.truetype(font_file, int(fontSize + 10))
    draw = ImageDraw.Draw(image)
    # output character
    char_ = rndChar()
    draw.text((20, -10), char_, font=font, fill=rndColor(1))
    ch = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return ch, char_


def persppective_transform(img):
    rows, cols, ch = img.shape
    scale1 = random.uniform(0.7, 0.9)
    scale2 = random.uniform(0.7, 0.95)
    scale3 = random.uniform(0.1, 0.4)
    p1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    # add diversity of the mode
    if random.uniform(0,1)>0.5:
        p2 = np.float32([[0, rows * (scale3)], [cols * scale1, rows * 0.1], [cols * 0.15, rows * scale2]])
    else:
        p2 = np.float32([[cols * (scale3), 0], [cols * 0.1, rows * scale1], [cols * scale2, rows * 0.15]])
    # p2 = np.float32([[0, rows * 0.1], [cols * 0.8, rows * 0.1], [cols * 0.15, rows * 0.7]])
    M = cv2.getAffineTransform(p1, p2)
    dst = cv2.warpAffine(img, M, (cols*2, rows*2))
    return dst


def get_real_wh(img):
    _img = copy.deepcopy(img)
    hmin, hmax, wmin, wmax = _img.nonzero()[0].min(), _img.nonzero()[0].max(), _img.nonzero()[1].min(), _img.nonzero()[
        1].max()
    _img = _img[hmin: hmax + 1, wmin:wmax + 1, :]
    w = wmax - wmin + 1
    h = hmax - hmin + 1
    return _img, int(w), int(h)


def conbine_img(new_img, ch, h_start, w_start):
    height, width, _ = ch.shape
    roi = new_img[h_start:h_start + height, w_start:w_start + width]
    img2gray = cv2.cvtColor(ch, cv2.COLOR_BGR2GRAY)  # convert to gray pic
    ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY_INV)  # backgroound white, foreground black
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    img2_fg = cv2.bitwise_and(ch, ch, mask=mask_inv)
    dst = cv2.add(img1_bg, img2_fg)
    new_img[h_start:h_start + height, w_start:w_start + width] = dst
    return new_img


if __name__ == '__main__':
    sample_num = 5
    dict_ = {
        "images": [],
        "annotations": [],
        "categories": [{'id': 1, 'name': 'A'},
                       {'id': 2, 'name': 'B'},
                       {'id': 3, 'name': 'C'},
                       {'id': 4, 'name': 'D'},
                       {'id': 5, 'name': 'E'}]
    }
    CLASSES = ['A', 'B', 'C', 'D', 'E']
    label2id = {cat: i for i, cat in enumerate(CLASSES)}
    box_id = 0
    os.makedirs('./combine_images', exist_ok=True)
    image_list = glob.glob('AllImages/*')

    img_list = random.sample(image_list, sample_num)

    # # 对图片进行合成
    for img_id, img_ in enumerate(tqdm(img_list)):
        # update img dict
        single_images_dict = {}

        # get sample num
        sample_num = random.randint(1, 5)

        # read img
        img = cv2.imread(img_)
        new_img_ = copy.deepcopy(img)  # clone
        # brightness
        new_img = imgBrightness(new_img_, 1.5, 3)

        # read character img and process them
        rec_mat = np.zeros((0, 4))
        for _ in range(sample_num):
            # update annotations dict
            single_annotations_dict = {
                'segmentation': [[]]
            }
            ch, char_ = get_character_img()
            # first augment
            chara_augmentation = pic_aug(0.9)
            ch = chara_augmentation(image=ch)['image']

            # then perspective
            ch = persppective_transform(ch)

            # get real wh
            ch, _, _ = get_real_wh(ch)

            # extra scale
            rescale_size = random.uniform(0.6, 1.3)
            ch = cv2.resize(ch, None, fx=rescale_size, fy=rescale_size, interpolation=cv2.INTER_AREA)
            ch_h, ch_w = ch.shape[0], ch.shape[1]
            img_h, img_w = img.shape[0], img.shape[1]
            h_start = random.randint(0, img_h - ch_h)
            w_start = random.randint(0, img_w - ch_w)
            if rec_mat.shape[0] == 0:
                rec_mat = np.concatenate((rec_mat, np.array([w_start, h_start, w_start + ch_w, h_start + ch_h])[None]))
            else:
                if not compute_iou(rec_mat, np.array([w_start, h_start, w_start + ch_w, h_start + ch_h])[None]):
                    continue
                else:
                    rec_mat = np.concatenate(
                        (rec_mat, np.array([w_start, h_start, w_start + ch_w, h_start + ch_h])[None]))

            # conbine img
            new_img = conbine_img(new_img, ch, h_start, w_start)
            # new_img[h_start:h_start + ch_h, w_start:w_start + ch_w, :] = ch

            # update box annotations
            single_annotations_dict['area'] = ch_h * ch_w
            single_annotations_dict['iscrowd'] = 0
            single_annotations_dict['image_id'] = img_id
            single_annotations_dict['bbox'] = [w_start, h_start, ch_w, ch_h]
            single_annotations_dict['id'] = box_id
            single_annotations_dict['category_id'] = label2id[char_] + 1
            dict_['annotations'].append(single_annotations_dict)
            # update box_id
            box_id += 1
        # save img
        cv2.imwrite(osp.join('combine_images', osp.basename(osp.splitext(img_)[0]) + '.jpg'), new_img)

        # update single_json information
        single_images_dict['file_name'] = osp.basename(osp.splitext(img_)[0]) + '.jpg'
        single_images_dict['height'] = img_h
        single_images_dict['width'] = img_w
        single_images_dict['id'] = img_id
        dict_['images'].append(single_images_dict)

    with open('train.json', 'w') as fp:
        json.dump(dict_, fp)

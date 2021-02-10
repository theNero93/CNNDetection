import argparse
import gc

import cv2
import dlib
import torch
import sys
import os

from tqdm import tqdm

from networks.resnet import resnet50

sys.path.append(os.getcwd())

from src.util.data.data_loader import load_data
from src.util.dotdict import Dotdict
from src.util.validate import calc_scores


def get_opt():
    opt = Dotdict()

    opt.model = 'all'
    opt.is_train = True
    opt.pretrained = True
    opt.checkpoints_dir = './out/checkpoints/faces'
    opt.continue_train = True
    opt.save_name = 'latest'
    opt.name = 'knn'
    # opt.dataset_path = './datasets/celeb-df-v2/faces'
    opt.dataset_path = './datasets/forensic/faces_best'
    opt.multiclass = False
    opt.resize_interpolation = 'bilinear'
    opt.load_size = -1
    opt.train_split = 'train'
    opt.train_size = 2500
    opt.val_split = 'val'
    opt.val_size = 100
    opt.test_split = 'test'
    opt.batch_size = 32

    return opt


def test(model_path, cuda):
    gc.collect()
    torch.cuda.empty_cache()
    print(cuda)
    model = resnet50(num_classes=1)
    if model_path is not None:
        state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.eval()
    if cuda:
        model.cuda()

    opt = get_opt()
    test_loader = load_data(opt, opt.test_split)
    y_pred, y_true = [], []
    for data, label in tqdm(test_loader):
        y_true.extend(label.flatten().tolist())
        if cuda:
            data = data.cuda()
        pred = model(data).sigmoid().flatten().tolist()
        y_pred.extend(pred)

    acc, ap, auc = calc_scores(y_true, y_pred)[:3]
    print("Test: acc: {}; ap: {}; auc: {}".format(acc, ap, auc))


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model_path', '-mi', type=str,
                   default='./src/baselines/cnn_detection/weights/blur_jpg_prob0.1.pth')
    p.add_argument('--cuda', type=bool, default=True)
    args = p.parse_args()
    test(**vars(args))

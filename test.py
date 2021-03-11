import argparse
import gc
from os.path import join

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from networks.resnet import resnet50
from src.data.basic_dataset import BasicDataset
from src.model.transforms.transform_builder import create_basic_transforms
from src.util.validate import calc_scores, save_pred_to_csv


def test(args):
    gc.collect()
    torch.cuda.empty_cache()
    print(args.cuda)
    model = resnet50(num_classes=1)
    state_dict = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.eval()
    if args.cuda:
        model.cuda()

    normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    dataset_dir = join(args.root_dir, args.dataset)
    test_data = BasicDataset(root_dir=dataset_dir,
                             processed_dir=args.processed_dir,
                             crops_dir=args.crops_dir,
                             split_csv=args.split_csv,
                             seed=args.seed,
                             normalize=normalize,
                             transforms=create_basic_transforms(args.size),
                             mode='test')
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    y_pred, y_true = [], []
    for data, label in tqdm(test_loader):

        if args.cuda:
            data = data.cuda()
        pred = model(data).sigmoid().flatten().tolist()
        y_true.extend(label.flatten().tolist())
        y_pred.extend(pred)

    acc, auc, loss = calc_scores(y_true, y_pred)[:3]
    print("Test: acc: {}; ap: {}; auc: {}".format(acc, auc, loss))
    if args.save_pred:
        save_pred_to_csv(y_true, y_pred, args.name, args.dataset)


def parse_args():
    parser = argparse.ArgumentParser(description="Parameters for Training")
    args = parser.add_argument
    # Dataset Options
    args("--root_dir", default='/bigssd/datasets', help="root directory")
    args("--dataset", default='dfdc', help="root directory")
    args('--processed_dir', default='processed', help='directory where the processed files are stored')
    args('--crops_dir', default='crops', help='directory of the crops')
    args('--split_csv', default='folds.csv', help='Split CSV Filename')
    args('--seed', default=111, help='Random Seed')
    args('--size', default=255)
    args('--name', default='bl_cnn_detection')
    args('--save_pred', default=True)
    args('--batch_size', default=32)

    # Model Options
    args('--model_path', '-mi', type=str, default='./src/baselines/cnn_detection/weights/blur_jpg_prob0.1.pth')
    args('--cuda', type=bool, default=True)
    return parser.parse_args()


if __name__ == '__main__':
    test(parse_args())

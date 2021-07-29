r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import time

import torch
from torch.nn.modules.activation import Threshold
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
# import torchvision.datasets.voc

from coco_utils import get_coco, get_coco_kp
from voc_utils import get_voc
from exdark_utils import get_ExDark
from car_utils import CarDetectionOnlyImage, get_Car

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, coco_evaluate, voc_evaluate, ekdark_evaluate, car_evaluate

import presets
import utils
import cv2
import glob
from iou import *

def get_dataset(name, image_set, transform, data_path, download=False):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2),
        "voc": (data_path, get_voc, 21),
        "ExDark": (data_path, get_ExDark, 13),
        "Car": (data_path, get_Car, 6)
    }
    p, ds_fn, num_classes = paths[name]

    if name == "voc":
        # 데이터셋 다운로드를 위함
        ds = ds_fn(p, image_set=image_set, transforms=transform, download=download)
    else:
        ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train, data_augmentation):
    return presets.DetectionPresetTrain(data_augmentation) if train else presets.DetectionPresetEval()


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Training', add_help=add_help)

    parser.add_argument('--data-path', default=os.path.abspath("../../Desktop/ai_data"), help='dataset') # modify # 'C:/Users/me1/Desktop/ai_data'
    parser.add_argument('--dataset', default='Car', help='dataset') # modify
    parser.add_argument('--model', default='retinanet_resnet50_fpn', help='model') # fasterrcnn_resnet50_fpn
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.0025, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu') # 0.0025 default
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-scheduler', default="multisteplr", help='the lr scheduler (default: multisteplr)')
    parser.add_argument('--lr-step-size', default=8, type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-steps', default=[15, 30], nargs='+', type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)') # default 16, 22 modify
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma (multisteplr scheduler only)')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default=os.path.abspath("../../Desktop/weights"), help='path where to save') # modify
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers', default=3, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument('--data-augmentation', default="hflip", help='data augmentation policy (default: hflip)')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--visualize-only",
        dest="visualize_only",
        help="Only visualize the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument(
        "--valid-only-img",
        dest="valid_only_img",
        help="Validation only image",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    return parser

# ///////////////////////////////////// main
def main(args):
    save_model_pth_name = "model_car"
    save_checkpoint_pth_name = "checkpoint_car"
    
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")
    
    # voc만 download 지원 (coco는 다운로드 불가)
    dataset, num_classes = get_dataset(args.dataset, "train", get_transform(True, args.data_augmentation),
                                       args.data_path, download=False)
    dataset_test, _ = get_dataset(args.dataset, "val", get_transform(False, args.data_augmentation),
                                  args.data_path, download=False)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    kwargs = {
        "trainable_backbone_layers": args.trainable_backbone_layers
    }
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes, pretrained=args.pretrained,
                                                              **kwargs)
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == 'multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosineannealinglr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError("Invalid lr scheduler '{}'. Only MultiStepLR and CosineAnnealingLR "
                           "are supported.".format(args.lr_scheduler))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # for g in optimizer.param_groups:
        #     g['lr'] = 0.0025
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        if 'coco' in args.dataset:
            coco_evaluate(model, data_loader_test, device=device)
        elif 'voc' in args.dataset:
            voc_evaluate(model, data_loader_test, device=device)
        elif 'ExDark' in args.dataset:
            ekdark_evaluate(model, data_loader_test, device=device)
        elif "Car" in args.dataset:
            car_evaluate(model, data_loader_test, device=device)
        return

    if args.visualize_only: # python train.py --resume model_25.pth --visualize-only
        model.eval()
        cpu_device = torch.device("cpu")
        
        threshold = 0.6
        skip_image_count = 0
        check_image_count = skip_image_count + 10
        # class_names = ("__background__", "aeroplane", "bicycle", "bird", "boat", "bottle", 
        # "bus", "car", "cat", "chair", "cow", 
        # "diningtable", "dog", "horse", "motorbike", "person", 
        # "pottedplant", "sheep", "sofa", "train", "tvmonitor" )
        
        # class_names = ("__background__", "Bicycle", "Boat",
        #     "Bottle", "Bus", "Car", "Cat", "Chair",
        #     "Cup", "Dog", "Motorbike", "People", "Table",)

        class_names = ("Unknown", "Car", "Bike", "Bus", "Truck", "Etc_vehicle")
      
        cnt = 0
        with torch.no_grad():
            if args.valid_only_img:
                print("Create Validation Image Dataset Car")
                images_valid_path = sorted(glob.glob(os.path.join(args.data_path, "Car_Data", "test", "*")))
                dataset_valid = CarDetectionOnlyImage(img_folder=args.data_path, all_images_path=images_valid_path, 
                    image_set="val", transforms=get_transform(True, args.data_augmentation))
                
                valid_sampler = torch.utils.data.SequentialSampler(dataset_valid)
                data_loader_valid = torch.utils.data.DataLoader(
                    dataset_valid, batch_size=1,
                    sampler=valid_sampler, num_workers=args.workers,
                    collate_fn=utils.collate_fn)

                for images, filenames in data_loader_valid:
                    images = list(img.to(device) for img in images)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    outputs = model(images)
                    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

                    for img, filename, output in zip(images, filenames, outputs):
                        img = img.cpu().numpy().transpose(1, 2, 0).copy()

                        indexes = indexing_removal_with_iou(output)

                        for i, (point, score, label) in enumerate(zip(output["boxes"], output["scores"], output["labels"])):
                            if score < threshold: continue
                            if i in indexes: continue
                            point = point.type(torch.IntTensor).numpy()
                            # x, y / xmax, ymax
                            img = cv2.rectangle(img, (point[0], point[1]), (point[2], point[3]), (0, 0, 255), 2)
                            img = cv2.putText(img, class_names[label], (point[0] + 2, point[1] - 9), 0, 0.5, (0, 0, 255), 2)
                            img = cv2.putText(img, str(round(score.item(), 2)), (point[0] + 2, point[1] + 9), 0, 0.4, (255, 0, 0), 2)

                        if skip_image_count <= cnt:
                            cv2.imshow(f"red: prediction / threshold: {threshold} / {filename}", img)
                            cv2.waitKey()
                            cv2.destroyAllWindows()

                        cnt += 1
                        if cnt == check_image_count:
                            exit()    
            else:
                for images, targets in data_loader_test:
                    images = list(img.to(device) for img in images)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    outputs = model(images)
                    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                    targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]

                    # print(outputs, targets)

                    for img, output, target in zip(images, outputs, targets):
                        img = img.cpu().numpy().transpose(1, 2, 0).copy() # cv2 = BGR, PIL RGB
                        # c, w, h -> w, h, c  / transpose axis -> 0, 1, 2 -> 1, 2, 0

                        indexes = indexing_removal_with_iou(output)

                        for i, (point, score, label) in enumerate(zip(output["boxes"], output["scores"], output["labels"])):
                            if score < threshold: continue
                            elif i in indexes: continue
                            point = point.type(torch.IntTensor).numpy()
                            # x, y / xmax, ymax
                            img = cv2.rectangle(img, (point[0], point[1]), (point[2], point[3]), (0, 0, 255), 2)
                            img = cv2.putText(img, class_names[label], (point[0] + 2, point[1] - 9), 0, 0.5, (0, 0, 255), 2)
                            img = cv2.putText(img, str(round(score.item(), 2)), (point[0] + 2, point[1] + 9), 0, 0.4, (255, 0, 0), 2)

                        for point, label in zip(target["boxes"], target["labels"]):
                            point = point.type(torch.IntTensor).numpy()
                            img = cv2.rectangle(img, (point[0], point[1]), (point[2], point[3]), (0, 255, 0), 2)
                            img = cv2.putText(img, class_names[label], (point[0] + 2, point[1] - 11), 0, 0.5, (0, 255, 0), 2)

                        filename = ''.join([chr(i) for i in target['name'].tolist()])
                        cv2.imshow(f"red: prediction / green: label / threshold: {threshold} / {filename}", img)
                        cv2.waitKey()
                        cv2.destroyAllWindows()

                        cnt += 1
                        if cnt == check_image_count:
                            exit()
        exit()

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch
            }
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, '%s.pth' % (save_model_pth_name + "_" + str(epoch))))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, '%s.pth' % save_checkpoint_pth_name))

        # evaluate after every epoch
        if 'coco' in args.dataset:
            coco_evaluate(model, data_loader_test, device=device)
        elif 'voc' in args.dataset:
            voc_evaluate(model, data_loader_test, device=device)
        elif 'ExDark' in args.dataset:
            ekdark_evaluate(model, data_loader_test, device=device)
        elif "Car" in args.dataset:
            if len(dataset_test) != 0:
                car_evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)

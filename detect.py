"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
import os
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from iou import *

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync


def parse_opt():
    default_path = os.path.join(os.path.expanduser('~'), 'Desktop/') # Desktop
    weights_path = "weights/face_track/"
    saved_pt = "best.pt"

    source_path = default_path + "ai_data/face_track/images/"
    ground_truth_path = default_path + "ai_data/face_track/labels/"

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=default_path + weights_path + saved_pt, help='model.pt path(s)') # default yolo5s.pt
    parser.add_argument('--source', type=str, default=source_path, help='file/dir/URL/glob, 0 for webcam') # default data/images
    parser.add_argument('--gt-source', type=str, default=ground_truth_path, help='ground truth sources') # ground truth source
    parser.add_argument('--imgset-dir', type=str, default="", help='image set directory') # imageset directory
    parser.add_argument('--show-gt', action='store_true', help='visualize ground_truth')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=default_path, help='save results to project/name') # default runs/detect
    parser.add_argument('--name', default='ai_data/face_track/visualize', help='save results to project/name') # default exp
    parser.add_argument('--exist-ok', action='store_false', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--show-image-count', default=[-1, 0], nargs='+', type=int,
                        help='number of show image count and number of skip image count (-1 0 is show all)') # default 16, 22 modify
    parser.add_argument('--add-pred-labels', action='store_true', help='add prediction labels in origin labels')
    opt = parser.parse_args()
    return opt


@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        gt_source='data/labels',
        imgset_dir='test',
        show_gt=False,
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        show_image_count=[-1, 0],
        add_pred_labels = False
        ):
    
    source += imgset_dir

    if show_gt:
        gt_source += imgset_dir
    else:
        gt_source = None
    
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, pt, onnx = False, w.endswith('.pt'), w.endswith('.onnx')  # inference type
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    if len(show_image_count) > 2 or len(show_image_count) < 2:
        print("show_image_count_list out of index. count was set default [-1, 0]")
        show_image_count = [-1, 0]

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, gt_source, img_size=imgsz, stride=stride)
        if show_image_count[1] > 0:
            print(f"Skip {show_image_count[1]} images to show")
            if dataset.nf < show_image_count[1]:
                print(f"Exception: Skip count({show_image_count[1]}) larger than files count({dataset.nf})")
                exit()
            dataset.count = show_image_count[1] # skip image count
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    show_all_image = False
    if show_image_count[0] <= 0:
        show_all_image = True

    check_image_count = show_image_count[0]
    cnt = 0

    # get image ---------------------------------------------------------------------------------------------------------
    t0 = time.time()
    for path, img, im0s, vid_cap, targets in dataset:
        if pt:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        elif onnx:
            img = img.astype('float32')
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        # Inference
        t1 = time_sync()
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            if show_gt and add_pred_labels:
                predict_label_num = 0
                det = det[(det[:, 5:6] == predict_label_num).any(1)]

            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            # im0 = cv2.copyMakeBorder(im0, height_pad, height_pad, 
            #     width_pad, width_pad, cv2.BORDER_CONSTANT, value=pad_color)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                # padding 추가
                height_pad = 30
                width_pad = 30
                pad_color = [0, 0, 0]

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                if dataset.mode == "image":
                    det[:, 0] += width_pad
                    det[:, 2] += width_pad
                    det[:, 1] += height_pad
                    det[:, 3] += height_pad

                    if targets is not None:
                        # target[:, :4] = scale_coords(img.shape[2:], target[:, :4], im0.shape).round()
                        targets[:, 0] += width_pad
                        targets[:, 2] += width_pad
                        targets[:, 1] += height_pad
                        targets[:, 3] += height_pad

                    im0 = cv2.copyMakeBorder(im0, height_pad, height_pad, 
                    width_pad, width_pad, cv2.BORDER_CONSTANT, value=pad_color)

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # will_remove_index = indexing_removal_with_iou(det)

                for i, (*xyxy, conf, cls) in enumerate(reversed(det)):
                    # add model prediction labels in origin files
                    if show_gt and add_pred_labels:
                        xyxy[0] = xyxy[0] - width_pad
                        xyxy[1] = xyxy[1] - height_pad
                        xyxy[2] = xyxy[2] - width_pad
                        xyxy[3] = xyxy[3] - height_pad

                        label_number = float(2)
                        cls = torch.tensor(label_number).to("cuda:0")

                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        line_count = 0
                        with open(gt_source + "/" + txt_path.split("\\")[-1] + ".txt", 'r') as ff:
                            for l in ff:
                                line_count += 1
                                break

                        with open(gt_source + "/" + txt_path.split("\\")[-1] + ".txt", 'a') as f:
                                if line_count != 0: f.write("\n")
                                f.write(('%g ' * len(line)).rstrip() % line)

                    # if i in will_remove_index: continue  # apply iou
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        # conf 이미지 신뢰도, hide_conf 이미지 신뢰도 표시 X
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        xyxy[0]
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                show_info_in_title = f"conf thres: {conf_thres} "
                if show_gt: show_info_in_title += "|show_ground_truth|*"
                if not hide_labels or not hide_conf:
                    if not hide_labels and not hide_conf:
                        show_info_in_title += "|show label and conf| "
                    elif not hide_labels and hide_conf:
                        show_info_in_title += "|show label| "
                
                if show_gt and targets is not None:
                    for t in targets:
                        t = t.type(torch.IntTensor).numpy()
                        im0 = cv2.rectangle(im0, (t[0], t[1]), (t[2], t[3]), (0, 255, 0), 0, cv2.LINE_AA)
                        im0 = cv2.putText(im0, names[t[5]], (t[0] + 2, t[1] - 9), 0, 0.5, (0, 0, 255), 2)
                cv2.imshow(show_info_in_title + str(p), im0)
                if dataset.mode == 'image':
                    k = cv2.waitKey()  # default 1 millisecond
                    cv2.destroyAllWindows()
                else:
                    k = cv2.waitKey(1)
                if k == ord('q'):   # q to quit
                    print("Exit")
                    exit()

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)
            
            if show_all_image == False:
                cnt += 1
                if cnt == check_image_count:
                    print("Exit")
                    exit()

    # end get image -----------------------------------------------------------------------------------------------------

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

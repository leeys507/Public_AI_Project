import sys
from utils.d_id import de_identification
sys.path.insert(0, './yolov5')

from utils.google_utils import attempt_download
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, tracking_id_check, xyxy2xywh, colorstr
from utils.torch_utils import select_device, time_sync
from utils.plots import colors, plot_one_box
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from iou import *
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate, \
        show_image_count, conf_thres, show_gt, hide_labels, hide_conf, gt_source, imgset_dir = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate, opt.show_image_count, opt.conf_thres, opt.show_gt, opt.hide_labels, opt.hide_conf, \
                opt.gt_source, opt.imgset_dir
    
    source += imgset_dir

    if show_gt:
        gt_source += imgset_dir
    else:
        gt_source = None
    
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    if len(show_image_count) > 2 or len(show_image_count) < 2:
        print("show_image_count_list out of index. count was set default [-1, 0]")
        show_image_count = [-1, 0]
    
    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, gt_source, img_size=imgsz)
        if show_image_count[1] > 0:
            print(f"Skip {show_image_count[1]} images to show")
            if dataset.nf < show_image_count[1]:
                print(f"Exception: Skip count({show_image_count[1]}) larger than files count({dataset.nf})")
                exit()
            dataset.count = show_image_count[1] # skip image count

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    show_all_image = False
    if show_image_count[0] <= 0:
        show_all_image = True

    check_image_count = show_image_count[0]
    cnt = 0
    tracking_id = [-1]

    # get image ---------------------------------------------------------------------------------------------------------
    for frame_idx, (path, img, im0s, vid_cap, targets) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_sync()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                # padding 추가
                height_pad = 30
                width_pad = 30
                pad_color = [0, 0, 0]

                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

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
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs = deepsort.update(dataset.mode, xywhs.cpu(), confs.cpu(), clss, im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    all_bboxes = [x[0:4] for x in outputs]
                    all_clss = [x[5] for x in outputs]
                    pair_fp, will_remove_indexes = indexing_person_with_intersection(all_bboxes, all_clss)
                    for j, (output, conf) in enumerate(zip(outputs, confs)): 
                        if j in will_remove_indexes: continue  # apply iou
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class

                        if c == 2:  # class == 2 (person)
                            if j in pair_fp:
                                face_label = names[pair_fp[j]]
                                visualize_name = f'{face_label}, ' + names[c]
                                label_color = pair_fp[j]
                                label = None if hide_labels else (names[c] if hide_conf else f'{id} {visualize_name} {conf:.2f}')
                                plot_one_box(bboxes, im0, label=label, color=colors(label_color + 1, True),
                                             line_thickness=2)
                        else:
                            visualize_name = names[c]
                            label_color = c
                            label = None if hide_labels else (names[c] if hide_conf else f'{id} {visualize_name} {conf:.2f}')
                            plot_one_box(bboxes, im0, label=label, color=colors(label_color + 1, True), line_thickness=2)

                        if tracking_id_check(tracking_id) and id not in tracking_id:
                            im0 = de_identification(im0, bboxes[0], bboxes[1], bboxes[2], bboxes[3])

                        if save_txt:
                            # to MOT format
                            bbox_top = output[0]
                            bbox_left = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                               f.write(('%g ' * 10 + '\n') % (frame_idx, id, bbox_top,
                                                           bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if show_vid:
                show_info_in_title = f"conf thres: {conf_thres} "
                if show_gt: show_info_in_title += "|show_ground_truth|*"
                if not hide_labels or not hide_conf:
                    if not hide_labels and not hide_conf:
                        show_info_in_title += "|show label and conf| "
                    elif not hide_labels and hide_conf:
                        show_info_in_title += "|show label| "
                    else:
                        show_info_in_title += "|show conf| "
                
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
                elif k == ord('i'):
                    tracking_id = [int(x) for x in input("Input Tracking ID list(0 or less is Show All)\n(Separated by white space): ").split()]

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

        if show_all_image == False:
            cnt += 1
            if cnt == check_image_count:
                print("Exit")
                exit()

    # end get image -----------------------------------------------------------------------------------------------------

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    default_path = os.path.join(os.path.expanduser('~'), 'Desktop/') # Desktop
    weights_path = "weights/face_track/"
    saved_pt = "best.pt"

    source_path = default_path + "ai_data/face_track/images/"
    ground_truth_path = default_path + "ai_data/face_track/labels/"

    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', type=str, default=default_path + weights_path + saved_pt, help='model.pt path(s)') # default yolo5s.pt
    parser.add_argument('--deep-sort-weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default=source_path, help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument('--show-image-count', default=[-1, 0], nargs='+', type=int,
                        help='number of show image count and number of skip image count (-1 0 is show all)') # default 16, 22 modify
    parser.add_argument('--show-gt', action='store_true', help='visualize ground_truth')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--gt-source', type=str, default=ground_truth_path, help='ground truth sources') # ground truth source
    parser.add_argument('--imgset-dir', type=str, default="", help='image set directory') # imageset directory
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    print(colorstr('tracking: ') + ', '.join(f'{k}={v}' for k, v in vars(parser.parse_args()).items()))
    with torch.no_grad():
        detect(args)

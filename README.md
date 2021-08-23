<div align="center">
<p>
<img width="850" src="face_detection.png"></a>
</p>
<br>
<div>
mask face and non mask face detection<br>
YOLOv5m pretrained model used<br>
add show ground truth for inference<br>
add hflip augmentation<br>
default_path is "absoulte_path/Desktop/"<br>
save model in default_path/weights<br>
</div>

### Pretrained Checkpoints

[assets]: https://github.com/ultralytics/yolov5/releases

|Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 |mAP<sup>val<br>0.5 |Speed<br><sup>V100 (ms) | |params<br><sup>(M) |FLOPs<br><sup>640 (B)
|---                    |---  |---      |---      |---      |---     |---|---   |---
|[YOLOv5s][assets]      |640  |36.7     |36.7     |55.4     |**2.0** |   |7.3   |17.0
|[YOLOv5m][assets]      |640  |44.5     |44.5     |63.1     |2.7     |   |21.4  |51.3
|[YOLOv5l][assets]      |640  |48.2     |48.2     |66.9     |3.8     |   |47.0  |115.4
|[YOLOv5x][assets]      |640  |**50.4** |**50.4** |**68.8** |6.1     |   |87.7  |218.8
|                       |     |         |         |         |        |   |      |
|[YOLOv5s6][assets]     |1280 |43.3     |43.3     |61.9     |**4.3** |   |12.7  |17.4
|[YOLOv5m6][assets]     |1280 |50.5     |50.5     |68.7     |8.4     |   |35.9  |52.4
|[YOLOv5l6][assets]     |1280 |53.4     |53.4     |71.1     |12.3    |   |77.2  |117.7
|[YOLOv5x6][assets]     |1280 |**54.4** |**54.4** |**72.0** |22.4    |   |141.8 |222.9
|                       |     |         |         |         |        |   |      |
|[YOLOv5x6][assets] TTA |1280 |**55.0** |**55.0** |**72.0** |70.8    |   |-     |-
<br>
<br>

### Inference
```cmd
python detect.py --conf-thres 0.6 --view-img --show-image-count 10 0 --imgset-dir test --show-gt --nosave
```
```cmd
python detect.py --conf-thres 0.6 --source https://www.youtube.com/watch?v=Ci47VF0v1pE --view-img  --nosave --show-image-count -1 0
```
<br>
<br>

### Tracking
```cmd
python track.py --conf-thres 0.7 --source https://www.youtube.com/watch?v=Pyx59BRQtOM --show-vid
```
```cmd
python track.py --conf-thres 0.7 --source https://www.youtube.com/watch?v=Pyx59BRQtOM --show-vid --classes 0 2 --blur-nontracking
```

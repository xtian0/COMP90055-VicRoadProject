# keras-yolo3

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/) and put into current directory.  
(change "yolov3.weights" to "yolov3-spp.weights" and "yolov3.cfg" to "yolov3-spp.cfg" if using spp model)
2. Convert the Darknet YOLO model to a Keras model.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py config/yolov3.cfg yolov3.weights model_data/yolo.h5
python convert.py -w config/yolov3.cfg yolov3.weights model_data/yolo_weights.h5
```

---

## Training

1. Put all images and xml files in:  
    * `model_data/pic`
2. Generate your own annotation file:
   
    * `python vicRoads_annotation.py`
    
    You will get an annotation file:
    * `model_data/train.txt` (for individual classes)
 
    Here is an example of annotation file:  
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

3. Make sure the variables: `annotation_path`, `log_dir`, and `classes_path` are correct in `train.py`
    
    Example: (line 18-20 in `train.py`)
    ```
    annotation_path = 'model_data/train.txt'
    log_dir = 'logs/'
    classes_path = 'model_data/vicRoads.names'
    ```
    (`log_dir` specifies the path to store the model trained)

    Start training:
    ```
    python train.py
    ```
    (Note: reduce the batch size in `train.py` (line 60 and 89) for stage 1 & 2 if run out of memory)

4. If you want to change the classes names, you will need to modify the class names in the following files:  
   
   * `vicRoads_annotation.py`
   * `model_data/vicRoads.names`


## Visualising loss
1. Training history will be stored in: 
   * `logs/hist_stage_1.pickle`
   * and `logs/hist_stage_2.pickle`
2. To show the loss plot:
```
python visualise.py --hist_dir=logs/
```
---

## Inferencing (image)
1. Put the test images in `test_imgs/`
2. Start testing
```
python test_picture.py

```
output pictures and files will be in `test_out`


## Inferencing (video)
1. Put the video for detection and counting in `test/`
2. Start inferencing
```
python test_video.py --model=logs/trained_weights_final.h5 \
                       --classes=model_data/vicRoads.names \
                       --gpu_num=1 \
                       --input=test/input_video_name.mkv \
                       --output=output/test_video.mkv
```
Output video will be in `output/`



## Evaluations (mAP)
Follow the instructions in `COMP90055-VicRoadProject/mAP/README.md` to generate the ground-truth files and run evaluations  


# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

# Deep sort
from counting.recorder import Recorder
from counting.deep_sort import preprocessing
from counting.deep_sort import nn_matching
from counting.deep_sort.detection import Detection
from counting.deep_sort.tracker import Tracker
from counting.tools import generate_detections as gdet

class YOLO(object):
    _defaults = {
        "model_path": 'logs/001/trained_weights_final.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/vicRoads.names',
        "score" : 0.4,                   # confidence threshold
        "iou" : 0.4,                     # iou for non-maximal suppression (NMS)
        "model_image_size" : (608, 608), # depends on pre-trained yolo model
        "gpu_num" : 1
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        # Setup default values and update with user overrides
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)

        # detection model
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

        # deep sort model
        self.tracker, self.encoder = self._load_deep_sort()
        self.records = {}
        self.nms_max_overlap = self.iou  # use same iou threshold for yolo and deepsort

        # init counts (class)
        self.counts = {i:j for (i,j) in zip(self.class_names, [0] * len(self.class_names))}
        
        # init direction counts
        possible_directions = ["NS", "SN", "ES", "WS", 
                               "NE", "SE", "EN", "WE", 
                               "NW", "SW", "EW", "WN"]
        self.directions = {i:j for (i,j) in zip(possible_directions, [0] * len(possible_directions))}

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        '''
            Load yolo detection model
        '''
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting

        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num >=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)

        # filtered bounding boxes with confidence threshold and NMS in each class 
        # (will ignore bus in yolo_eval)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)

        return boxes, scores, classes

    def _load_deep_sort(self):
        # load deep sort model
        model_filename = 'counting/data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        max_cosine_distance = 0.4
        nn_budget = None
        # either cosine or euclidean distance
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 
                                                           max_cosine_distance, 
                                                           nn_budget)
        tracker = Tracker(metric)
        return tracker, encoder

    def detect_image(self, image, frames):
        
        # start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)

        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # Get outputs from pre-defined 
        # (defined in yolo_eval) output tensor targets
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # tlbr boxes to tlwh boxes
        tlwh_boxes = []
        for box in out_boxes:
            # tlwh
            tlwh_boxes.append([box[0], box[1], 
                               box[3] - box[1], box[2] - box[0]])

        # Get features for tracking (takes tlwh boxes)
        features = self.encoder(np.array(image), tlwh_boxes)

        # Get detection objects
        detections = [Detection(bbox, score, feature, label) for bbox, feature, label, score in 
                      zip(tlwh_boxes, features, out_classes, out_scores)]

        # Run non-maxima suppression
        tlwh_boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(tlwh_boxes, 
                                                    self.nms_max_overlap, 
                                                    scores)
        detections = [detections[i] for i in indices]
        
        detections = self.handle_DGV_sign(detections)

        print('Found {} boxes for {}'.format(len(detections), 'img'))

        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # drawing setting
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 400

        # label
        for track in self.tracker.tracks:

            if track.track_id not in self.records:
                self.records[track.track_id] = Recorder(track.track_id, 
                                                        track.label, 
                                                        frames)
            else:
                self.records[track.track_id].update(frames)
            if track.is_confirmed() and track.time_since_update > 1:
                continue

            # increment count only if the track have 10 or more frames
            if self.records[track.track_id].num_frames() == 10:
                name = self.class_names[self.records[track.track_id].label]
                self.counts[name] += 1

            bbox = track.to_tlbr()
            top, left, bottom, right = bbox
            
            # identified as DGV if the track has 10 or more DGV sign detected
            if track.DGV_count >= 10:
                label = 'ID{}: {} (DGV)'.format(track.track_id, 
                                        self.class_names[track.label])
            else:
                label = 'ID{}: {}'.format(track.track_id, 
                                        self.class_names[track.label])
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # label
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[track.label])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)

            # bounding box
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[track.label])
                
        # identify direction
        for track in self.tracker.just_deleted:
            direction_name = self.get_direction(track)
            print(direction_name)
            if direction_name != "NA":
                self.directions[direction_name] += 1
        
        
        # draw counts (class) text
        draw = ImageDraw.Draw(image)
        count_origin = np.array([10, image.height - 290])
        count_str = "Counts: \n"
        changeLine = 0
        
        # -2 for excluding van and bus
        for class_name, count in list(self.counts.items())[:-2]:
            
            count_str += "{}: {}   ".format(class_name, count)

            changeLine += 1
            # change line
            if changeLine == 3: 
                count_str += "\n"
                changeLine = 0
        draw.text(count_origin, count_str, fill=(255, 255, 255), font=font)
        
        
        # draw direction counts text
        direction_origin = np.array([image.width - 270, image.height - 200])
        direction_str = "Direction counts: \n"
        changeLine = 0
        for direction_name, count in self.directions.items():
            
            direction_str += "{}: {}   ".format(direction_name, count)

            changeLine += 1
            # change line
            if changeLine == 2: 
                direction_str += "\n"
                changeLine = 0
        draw.text(direction_origin, direction_str, 
                  fill=(255, 255, 255), font=font)
        
        del draw

        # end = timer()

        # print(end - start)

        return image

    def close_session(self):
        self.sess.close()
        
    def handle_DGV_sign(self, detections):
        
        DGV_detections = []
        other_detections = []
        
        for detection in detections:
            if self.class_names[detection.label] == "DGV":
                DGV_detections.append(detection)
            else:
                other_detections.append(detection)
                
        # no DGV sign detected
        if len(DGV_detections) == 0:
            return other_detections
        
        # otherwise, assign DGV sign to other detections
        for dgv in DGV_detections:
            
             top_d, left_d, bottom_d, right_d = dgv.to_tlbr()
             
             for detection in other_detections:
                top, left, bottom, right = detection.to_tlbr()
                
                # DGV sign inside detection box
                if top_d >= top and bottom_d <= bottom and \
                    left_d >= left and right_d <= right:
                        detection.DGV = True
        
        return other_detections
        
        
    # Function for identifying the direction by pre-defined regions
    def get_direction(self, track):
        
        # get center of the trucks at start and end of tracks
        center_list = track.centers
        
        # start center
        x0 = center_list[0][0]
        y0 = center_list[0][1]
        
        # end center
        x9 = center_list[-1][0]
        y9 = center_list[-1][1]
        
        print("start:")
        print(x0, y0)
        print("end:")
        print(x9, y9)
        
        # start region
        if x0 < 1100 and y0 < 500:
            start = "W"
        elif x0 > 1200 and y0 < 500:
            start = "N"
        elif x0 > 1300 and y0 > 600:
            start = "E"
        elif 300 < x0 < 1300 and y0 > 600:
            start = "S"
        else:
            # can't find direction
            return "NA"
        
        # end region
        if x9 < 700 and y9 > 300:
            end = "W"
        elif 1800 > x9 > 1000 and y9 < 350:
            end = "N"
        elif x9 > 1500  and 300 < y9 < 600:
            end = "E"
        elif x9 > 1300 and y9 > 600:
            end = "S"
        else:
            # can't find direction
            return "NA"
        
        # output
        if start == end:
            return "NA"
        else:
            return start + end

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    video_FourCC = cv2.VideoWriter_fourcc('M','P','4','2')
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), 
              type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    frame_num = 0

    while True:
        # read video
        return_value, frame = vid.read()
        if not return_value:
            break

        # detect image and get results
        image = Image.fromarray(frame)
        image = yolo.detect_image(image, frame_num)
        result = np.asarray(image)

        frame_num += 1

        # count time and fps
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        
        # show fps
        cv2.putText(result, text=fps, org=(3, 15), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)

        # write output video
        if isOutput:
            out.write(result)
    yolo.close_session()
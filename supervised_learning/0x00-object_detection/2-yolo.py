#!/usr/bin/env python3
"""contains class Yolo"""
import tensorflow.keras.backend as backend
import tensorflow as tf
import tensorflow.keras as K
from numpy import concatenate as cat
from numpy.core.fromnumeric import searchsorted
import numpy as np


class Yolo():
    """creating class Yolo"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """initializing class Yolo"""
        self.model = K.models.load_model(model_path)
        class_names = []
        with open(classes_path, 'r') as file:
            for name in file:
                class_names.append(name.strip())
        self.class_names = class_names
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, arr):
        """sigmoid activation function"""
        return 1 / (1+np.exp(-1*arr))

    def process_outputs(self, outputs, image_size):
        """proccesses output"""
        IH, IW = image_size[0], image_size[1]
        boxes = [output[..., :4] for output in outputs]
        box_confidence, class_probs = [], []
        cornersX, cornersY = [], []

        for output in outputs:
            # Organize grid cells
            gridH, gridW, anchors = output.shape[:3]
            cx = np.arange(gridW).reshape(1, gridW)
            cx = np.repeat(cx, gridH, axis=0)
            cy = np.arange(gridW).reshape(1, gridW)
            cy = np.repeat(cy, gridH, axis=0).T

            cornersX.append(
                np.repeat(cx[..., np.newaxis], anchors, axis=2)
                )
            cornersY.append(
                np.repeat(cy[..., np.newaxis], anchors, axis=2)
                )
            # box confidence and class probability activations
            box_confidence.append(self.sigmoid(output[..., 4:5]))
            class_probs.append(self.sigmoid(output[..., 5:]))

        inputW = self.model.input.shape[1].value
        inputH = self.model.input.shape[2].value

        # Predicted boundary box
        for x, box in enumerate(boxes):
            bx = (
                (self.sigmoid(box[..., 0])+cornersX[x])/outputs[x].shape[1]
                )
            by = (
                (self.sigmoid(box[..., 1])+cornersY[x])/outputs[x].shape[0]
                )
            bw = (
                (np.exp(box[..., 2])*self.anchors[x, :, 0])/inputW
                )
            bh = (
                (np.exp(box[..., 3])*self.anchors[x, :, 1])/inputH
                )

            # x1
            box[..., 0] = (bx - (bw * 0.5))*IW
            # y1
            box[..., 1] = (by - (bh * 0.5))*IH
            # x2
            box[..., 2] = (bx + (bw * 0.5))*IW
            # y2
            box[..., 3] = (by + (bh * 0.5))*IH

        return (boxes, box_confidence, class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """creates filter boxes"""
        best_boxes, scores, classes = None, None, None
        for x in range(len(boxes)):
            box_score = box_confidences[x] * box_class_probs[x]
            box_class = np.argmax(box_score, axis=-1)
            box_score = np.amax(box_score, axis=-1)
            mask = box_score >= self.class_t

            if best_boxes is None:
                best_boxes = boxes[x][mask]
                scores = box_score[mask]
                classes = box_class[mask]
            else:
                best_boxes = cat((best_boxes, boxes[x][mask]), axis=0)
                scores = cat((scores, box_score[mask]), axis=0)
                classes = cat((classes, box_class[mask]), axis=0)

        return (best_boxes, classes, scores)

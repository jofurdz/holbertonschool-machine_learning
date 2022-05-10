#!/usr/bin/env python3
"""contains class Yolo"""
import tensorflow.keras.backend as backend
import tensorflow as tf
import tensorflow.keras as K
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

    def process_outputs(self, outputs, image_size):
        """Processses the output of the Darkent Model"""
        def sigmoid(array):
            """Sigmoid activation function"""
            return 1 / (1 + np.exp(-1 * array))

        boxes, box_confidences, box_class_probs = [], [], []
        image_width = self.model.input.shape[1]  # .value
        image_height = self.model.input.shape[2]  # .value

        for i, output in enumerate(outputs):
            output_boxes = output[..., :4]
            grid_height, grid_width, anchors = output.shape[:3]

            cx = np.arange(grid_width).reshape(1, grid_width)
            cx = np.repeat(cx, grid_height, axis=0)
            cx = np.repeat(cx[..., np.newaxis], anchors, axis=2)
            cy = np.arange(grid_width).reshape(1, grid_width)
            cy = np.repeat(cy, grid_height, axis=0).T
            cy = np.repeat(cy[..., np.newaxis], anchors, axis=2)

            tx = output_boxes[..., 0]
            ty = output_boxes[..., 1]
            tw = output_boxes[..., 2]
            th = output_boxes[..., 3]

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            bx = (sigmoid(tx) + cx) / grid_width
            by = (sigmoid(ty) + cy) / grid_height
            bw = (pw * np.exp(tw)) / image_width
            bh = (ph * np.exp(th)) / image_height

            x1 = (bx - (bw / 2)) * image_size[1]
            y1 = (by - (bh / 2)) * image_size[0]
            x2 = (bx + (bw / 2)) * image_size[1]
            y2 = (by + (bh / 2)) * image_size[0]

            output_boxes[..., 0] = x1
            output_boxes[..., 1] = y1
            output_boxes[..., 2] = x2
            output_boxes[..., 3] = y2

            boxes.append(output_boxes)
            box_confidences.append(sigmoid(output[..., 4:5]))
            box_class_probs.append(sigmoid(output[..., 5:]))

        return (boxes, box_confidences, box_class_probs)

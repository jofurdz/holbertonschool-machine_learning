#!/usr/bin/env python3
"""function for cre3ating class Yolo"""
import numpy as np
import tensorflow.keras as K


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

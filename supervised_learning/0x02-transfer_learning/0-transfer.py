#!/usr/bin/env python3
"""function for classifying the cifar10 dataset using transfer learning"""
import tensorflow.keras as K


def preprocess_data(X, Y):
    X_p = K.applications.mobilenet_v2.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y)
    return (X_p, Y_p)


if __name__ == '__main__':
    y_train = K.utils.to_categorical(y_train)
    (x_train, y_train), (x_valid, y_valid) = K.datasets.cifar10.load_data()
    x_trainPre, y_trainPre = preprocess_data(x_train, y_train)
    x_validPre, y_validPre = preprocess_data(x_valid, y_valid)

    mnv2 = K.applications.MobileNetV2(weights='imagenet', input_tensor=input,
                                      include_top=False, pooling='same',
                                      input_shape=(32, 32, 3))
    input = K.Input(shape=(32, 32, 3))

    layer = mnv2(input, training=False)
    layer = K.layers.GlobalAveragePooling2D()(layer)
    layer = K.layers.BatchNormalization()(layer)
    layer = K.layers.Dense(512, activation='relu')(layer)
    layer = K.layers.Dropout(0.2)(layer)
    layer = K.layers.Dense(512, activation='relu')(layer)
    layer = K.layers.Dropout(0.2)(layer)
    output = K.layers.Dense(10, activation='softmax')(layer)

    model = K.Model(inputs=input, outputs=output)
    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['acc'])
    model.fit(x_trainPre, y_trainPre,
              validation_data=(x_validPre, y_validPre),
              epochs=10, verbose=1)
    model.save('cifar10.h5')

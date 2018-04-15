from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

def shifting_sandwich_model():
    ways = [
        [Input(shape=(96,96,1))],
        [Input(shape=(96,96,1))],
        [Input(shape=(96,96,1))]
    ]

    best_2d = load_model('saved_models/model-babystep3-composite-tertiary-classification.h5')
    
    c1 = Convolution2D(32, 3, 3, activation='relu', weights=best_2d.layers[0].get_weights())
    c2 = Convolution2D(64, 3, 3, activation='relu', weights=best_2d.layers[2].get_weights())
    c3 = Convolution2D(128, 3, 3, activation='relu')
    c4 = Convolution2D(256, 3, 3, activation='relu')

    common_way = [
        c1,
        c2,
        MaxPooling2D(pool_size=(2,2)),
        BatchNormalization(),
        c3,
        c4,
        MaxPooling2D(pool_size=(2,2)),
        BatchNormalization()
    ]

    for i in range(3):
        for idx, layer in enumerate(common_way):
            ways[i].append(layer(ways[i][idx]))

    ways = np.array(ways)

    merged = merge(ways[:,-1], mode='concat')

    c5 = Convolution2D(512, 3, 3, activation='relu')(merged)
    c6 = Convolution2D(256, 3, 3, activation='relu')(c5)

    flatten = Flatten()(c6)
    
    d1 = Dense(1024, activation='relu')(flatten)
    d2 = Dense(512, activation='relu')(d1)
    d3 = Dense(3, activation='sigmoid')(d2)

    model = Model(input=list(ways[:,0]), output=d3)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['precision', 'recall'])
    
    return model

def shifting_sandwich_model_3d_conv():
    inputs = Input(shape=(3,96,96,1))
    
    MP2D = MaxPooling3D(pool_size=(1, 2, 2))
    DROP = Dropout(0.1)
    NORM = BatchNormalization()
    
    c1 = Convolution3D(16, 1, 3, 3, activation='relu')(inputs)
    c2 = Convolution3D(16, 1, 3, 3, activation='relu')(c1)
    
    c3 = Convolution3D(32, 1, 3, 3, activation='relu')(MP2D(c2))
    c4 = Convolution3D(64, 1, 3, 3, activation='relu')(MP2D(c3))
    
    c5 = Convolution3D(128, 3, 3, 3, activation='relu')(DROP(MP2D(c4)))
    c6 = Convolution3D(64, 1, 3, 3, activation='relu')(c5)

    flatten = Flatten()(NORM(MP2D(c6)))
    
    d1 = Dense(256, activation='relu')(flatten)
    d2 = Dense(64, activation='relu')(d1)
    d3 = Dense(4, activation='sigmoid')(d2)

    model = Model(input=inputs, output=d3)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.95, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['precision', 'recall'])
    
    return model
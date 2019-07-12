from net import inference
from datagen import SequenceData
import numpy as np
import math
from keras.callbacks import ModelCheckpoint


model_savepath = "weights.best.hdf5"
epoch = 20
batch_size = 4
x_index = np.load('x_index.npy')  #load data index
y_index = np.load('y_index.npy')  #load label index
sequence_data = SequenceData(x_index, y_index, batch_size)
steps = math.ceil(len(x_index) / batch_size)
myModel = inference()
model_checkpoint = ModelCheckpoint(model_savepath, monitor='loss', verbose=1, save_best_only=True)
myModel.fit_generator(sequence_data, steps_per_epoch=steps, epochs=epoch, verbose=1,
                    callbacks=[model_checkpoint],use_multiprocessing=False, workers=1)

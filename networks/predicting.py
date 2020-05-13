from net import resnet
import os
import numpy as np
from scipy import misc

testimg_path = "D:/LeeX/deep-learning-microscopy/20190410perspective_image_new_9x11/npydata/"
predimg_savepath = "D:/LeeX/deep-learning-microscopy/20190410perspective_image_new_9x11/predict_img/"
test_path = 'D:/LeeX/deep-learning-microscopy/test/input'
model_path = "weights.best.hdf5"

count_test=int(len(os.listdir(test_path))/2)

model = inference()
print('loading weights...')
model.load_weights(model_path)
print('overloaded data succeeded!')
print('predicting...')
for i in range(9):
    for j in range(11):
        test_img = np.load((testimg_path + 'input_data_(%d,%d).npy') % (i+1,j+1))
        test_img = np.reshape(test_img,[1,64,64,1])
        predimg = model.predict(test_img)
        predimg = np.squeeze(predimg)
        predimg = (predimg-np.min(predimg))/(np.max(predimg)-np.min(predimg))
        #predimg = predimg.astype('float32')/255
        misc.imsave((predimg_savepath+'(%d,%d).png') % (i+1,j+1), predimg)
        print(('predicted img(%d,%d) has saved!') % (i+1,j+1))

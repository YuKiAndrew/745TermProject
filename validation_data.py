from collections import Counter
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from imblearn.over_sampling import SMOTE
import time
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import  ADASYN

list11 = []  # data
list22 = []  # label_2
list1010 = []  # label_10
for serialized_example in tf.compat.v1.io.tf_record_iterator("normalized/validation_select_12.tfrecords"):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    feature = example.features.feature['features'].float_list.value
    label_2 = example.features.feature['label_2'].float_list.value
    label_10 = example.features.feature['label_10'].float_list.value

    list11.append(feature)
    list22.append(label_2)
    list1010.append(label_10)

XX=np.array(list11)
b22=np.array(list22)
bb22=b22.reshape(b22.shape[0],)
b1010=np.array(list1010)
bb1010=b1010.reshape(b1010.shape[0],)
y22 = np.int32(bb22)
y1010 = np.int32(bb1010)

print(y22.shape)
print(y1010.shape)
print(XX.shape)
print(Counter(y22))
print(Counter(y1010))

yy22 = y22.reshape(y22.shape[0],1)
print(yy22.shape)
yy1010 = y1010.reshape(y1010.shape[0],1)
print(yy1010.shape)

np.save("alldata/data_12_val.npy",XX)
np.save("alldata/label_2_12_val.npy",yy22)
np.save("alldata/label_10_12_val.npy",yy1010)
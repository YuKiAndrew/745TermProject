import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from collections import Counter
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from imblearn.over_sampling import SMOTE
import time
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import  ADASYN

# 读取tf文件数据
# Read the tf file data
list1 = []  # data
list2 = []  # label_2
list10 = []  # label_10

for serialized_example in tf.compat.v1.io.tf_record_iterator("normalized/train_select_12.tfrecords"):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    feature = example.features.feature['features'].float_list.value
    label_2 = example.features.feature['label_2'].float_list.value
    label_10 = example.features.feature['label_10'].float_list.value
    list1.append(feature)
    list2.append(label_2)
    list10.append(label_10)

# for serialized_example in tf.python_io.tf_record_iterator("/home/hll/IDS/normalized/test_select_12.tfrecords"):


#     example = tf.train.Example()
#     example.ParseFromString(serialized_example)


#     feature = example.features.feature['features'].float_list.value
#     label_2 = example.features.feature['label_2'].float_list.value
#     label_10 = example.features.feature['label_10'].float_list.value

#     list1.append(feature)
#     list2.append(label_2)
#     list10.append(label_10)


# for serialized_example in tf.python_io.tf_record_iterator("/home/hll/IDS/normalized/validation_select_12.tfrecords"):


#     example = tf.train.Example()
#     example.ParseFromString(serialized_example)


#     feature = example.features.feature['features'].float_list.value
#     label_2 = example.features.feature['label_2'].float_list.value
#     label_10 = example.features.feature['label_10'].float_list.value

#     list1.append(feature)
#     list2.append(label_2)
#     list10.append(label_10)

X=np.array(list1)
b2=np.array(list2)
bb2=b2.reshape(b2.shape[0],)
b10=np.array(list10)
bb10=b10.reshape(b10.shape[0],)
y2 = np.int32(bb2)
y10 = np.int32(bb10)


print(Counter(y2))
print(Counter(y10))

time_start = time.time()
smo = SMOTE(random_state=42)

X_smo, y_smo = smo.fit_resample(X, y2)
X_smo_10, y_smo_10 = smo.fit_resample(X, y10)

time_end = time.time()
time = time_end - time_start
print("time:",time)

sorted(Counter(y_smo).items())

yy2 = y_smo.reshape(y_smo.shape[0],1)
print(yy2.shape)
yy10 = y_smo_10.reshape(y_smo_10.shape[0],1)
np.save("alldata/12/tomektrain/data.npy",X_smo)
np.save("alldata/12/tomektrain/label_2.npy",yy2)
np.save("alldata/12/tomektrain/label_10.npy",yy10)
np.save("alldata/12/tomektrain/data_10.npy",X_smo_10)
print("tomktrain10_shape: ", yy10.shape)

list11 = []  # data
list22 = []  # label_2
list1010 = []  # label_10

# for serialized_example in tf.compat.v1.io.tf_record_iterator("normalized/validation_select_12.tfrecords"):
#     example = tf.train.Example()
#     example.ParseFromString(serialized_example)
#
#     feature = example.features.feature['features'].float_list.value
#     label_2 = example.features.feature['label_2'].float_list.value
#     label_10 = example.features.feature['label_10'].float_list.value
#
#     list11.append(feature)
#     list22.append(label_2)
#     list1010.append(label_10)
#
# XX=np.array(list11)
# b22=np.array(list22)
# bb22=b22.reshape(b22.shape[0],)
# b1010=np.array(list1010)
# bb1010=b1010.reshape(b1010.shape[0],)
# y22 = np.int32(bb22)
# y1010 = np.int32(bb1010)
#
# print(y22.shape)
# print(y1010.shape)
# print(XX.shape)
# print(Counter(y22))
# print(Counter(y1010))
#
# yy22 = y22.reshape(y22.shape[0],1)
# print(yy22.shape)
# yy1010 = y1010.reshape(y1010.shape[0],1)
# print(yy1010.shape)
#
# np.save("alldata/data_12_val.npy",XX)
# np.save("alldata/label_2_12_val.npy",yy22)
# np.save("alldata/label_10_12_val.npy",yy1010)

# it is from the train set
ros = RandomOverSampler(random_state=0)
X_ros, y_ros = ros.fit_resample(X, y2)


sorted(Counter(y_ros).items())

yy2 = y_ros.reshape(y_ros.shape[0],1)
yy2.shape

np.save("cicdata/77_2/data_ros_train.npy",X_ros)
np.save("cicdata/77_2/label_ros_train.npy",yy2)

ros = RandomOverSampler(random_state=0)
X_ros, y_ros = ros.fit_resample(X, y10)

sorted(Counter(y_ros).items())

yy10 = y_ros.reshape(y_ros.shape[0],1)
yy10.shape

np.save("alldata/12/rostrain/data_ros10.npy",X_ros)
np.save("alldata/12/rostrain/label_10_ros.npy",yy10)

smo = SMOTE(random_state=42)

X_smo, y_smo = smo.fit_resample(X, y2)
sorted(Counter(y_smo).items())

yy2 = y_smo.reshape(y_smo.shape[0],1)
yy2.shape

np.save("cicdata/77_2/data_smote_train.npy",X_smo)
np.save("cicdata/77_2/label_smote_train.npy",yy2)

smo = SMOTE(random_state=42)

X_smo, y_smo = smo.fit_resample(X, y10)
sorted(Counter(y_smo).items())

yy10 = y_smo.reshape(y_smo.shape[0],1)
yy10.shape
np.save("alldata/12/smotetrain/data_12smo10.npy",X_smo)
np.save("alldata/12/smotetrain/label_10_12smo.npy",yy10)

yy10.shape

X_adasyn, y_adasyn = ADASYN().fit_resample(X, y2)
print(sorted(Counter(y_adasyn).items()))

yy2 = y_adasyn.reshape(y_adasyn.shape[0],1)
yy2.shape
np.save("cicdata/77_2/data_adasyn_train.npy",X_adasyn)
np.save("cicdata/77_2/label_adasyn_train.npy",yy2)

X_adasyn, y_adasyn = ADASYN(sampling_strategy='minority',random_state=42).fit_resample(X, y10)

print(sorted(Counter(y_adasyn).items()))

yy2 = y_adasyn.reshape(y_adasyn.shape[0],1)
yy2.shape
np.save("alldata/12/ADASYN/data_12adasyn10.npy",X_adasyn)
np.save("alldata/12/ADASYN/label_10_12adasyn.npy",yy2)
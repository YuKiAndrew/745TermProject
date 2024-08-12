from collections import Counter
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
import time
from imblearn.under_sampling import RandomUnderSampler

data = np.load('alldata/12/train/data.npy')
label = np.load('alldata/12/train/label_2.npy')

X=np.array(data)
b=np.array(label)
bb=b.reshape(b.shape[0],)
y10 = np.int32(bb)

sorted(Counter(y10).items())


time_start = time.time()

smo = SMOTE(ratio={1:889015},random_state=42)
#smo = SMOTE(ratio={1:177803,2:177803,3:177803,4:177803,5:177803,6:177803,7:177803,8:177803,9:177803},random_state=42)

X_smo, y_smo = smo.fit_sample(X, y10)
print(sorted(Counter(y_smo).items()))

time_end = time.time()
time = time_end - time_start
print("time:",time)

X_smo.shape[0]

print("undersample")
time_start = time.time()

rus = RandomUnderSampler(ratio={0:889015},random_state=42)

X_rus, y_rus = rus.fit_sample(X_smo, y_smo)
print(sorted(Counter(y_rus).items()))

time_end = time.time()
time = time_end - time_start
print("time:",time)


label_end = y_rus.reshape(y_rus.shape[0],1)

np.save("alldata/12/RUS+SMOTE/label_2/data.npy",X_rus)
np.save("alldata/12/RUS+SMOTE/label_2/label.npy",label_end)
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import regularizers
import time
import keras

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics
from sklearn.metrics import classification_report


x_train = np.load('alldata/12/rostrain/data_ros10.npy')
y_train = np.load('alldata/12/rostrain/label_10_ros.npy')

x_test = np.load('alldata/data_12_test.npy')
y_test = np.load('alldata/label_10_12_test.npy')

x_val = np.load('alldata/data_12_val.npy')
y_val = np.load('alldata/label_10_12_val.npy')

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape,x_val.shape,y_val.shape)

def encode(data):
    print('Shape of data (BEFORE encode): %s' % str(data.shape))
    encoded = to_categorical(data)
    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
    return encoded

y_train_onehot = encode(y_train)
y_test_onehot = encode(y_test)
y_val_onehot = encode(y_val)

model = Sequential()
model.add(Dense(input_dim = 12,
                units = 128,
                activation = 'relu'))


model.add(Dense(units = 128,activation = 'relu'))

model.add(Dense(units = 64, activation = 'relu'))

model.add(Dense(units = 32, activation = 'relu'))

model.add(Dense(units = 10,activation = 'softmax'))

print(model.summary())

time_start = time.time()

callback_list = [keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=30, mode='max',),
                keras.callbacks.ModelCheckpoint(filepath='my_model.keras',monitor='val_accuracy',save_best_only=True,mode='max',)]

model.compile(loss = "categorical_crossentropy",optimizer = "nadam", metrics = ["accuracy"])
history = model.fit(x = x_train,y = y_train_onehot,
                epochs = 20,
                batch_size = 256,
                verbose = 2,
                callbacks=callback_list,
                validation_data=(x_val, y_val_onehot) )

time_end = time.time()
train_time = time_end - time_start

model.save('alldata/12/rostrain/MLP_RUS_10.h5')

scores = model.evaluate(x_test, y_test_onehot)
print("test_loss = ", scores[0],"test_accuracy = ", scores[1])

time_start = time.time()

y_pred_onehot  = model.predict(x_test)
y_pred_label=np.argmax(y_pred_onehot,axis=1)

time_end = time.time()
test_time = time_end - time_start
print("test_time:",test_time)

np.savetxt("alldata/12/rostrain/MLP_y_pred_10.txt",y_pred_label)

y_true_onehot=y_test_onehot
y_true_label=np.argmax(y_true_onehot,axis=1)
np.savetxt("alldata/12/MLP_y_true_10.txt",y_true_label)

labels = ['Normal', 'Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode',
          'Worms']

y_true = y_true_label
y_pred = y_pred_label

tick_marks = np.array(range(len(labels))) + 0.5


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 10), dpi=100)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.001:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='MLP_12_10_ROS Normalized confusion matrix')
plt.show()

target_names = ['Normal','Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic','Reconnaissance','Shellcode','Worms']
print(classification_report(y_true,y_pred,target_names=target_names))

acc = metrics.accuracy_score(y_true,y_pred)
f1 = metrics.f1_score(y_true, y_pred,average='weighted')
pre = metrics.precision_score(y_true, y_pred, labels=None, pos_label=1, average='weighted')  #DR
recall = metrics.recall_score(y_true, y_pred, labels=None, pos_label=1, average='weighted', sample_weight=None)


print("acc:",acc)
print("pre:",pre)
print("DR=recall:",recall)
print("f1:",f1)

TP=cm[1,1]
FP=cm[0,1]
FN=cm[1,0]
TN=cm[0,0]

print("TP:",TP)
print("FP:",FP)
print("FN:",FN)
print("TN:",TN)

acc = (TP+TN)/(TP+TN+FP+FN)
print("acc:",acc)

DR = TP/(TP+FN)
print("DR:",DR)

FPR = FP/(FP+TN)  #FAR
print("FPR:",FPR)

recall =TP/(TP+FN)
print("recall：",recall)

precision = TP/(TP+FP)
print("precision:",precision)

f1 = (2*precision*recall)/(precision+recall)
print("f1:",f1)
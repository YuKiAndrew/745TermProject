import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def make_tfrecords(dataset, file_to_save):

	try:
		data = dataset.values
	except:
		data = dataset
	with tf.io.TFRecordWriter(file_to_save) as writer:
		for rows in data:
			features, label_10, label_2 = rows[:-2], rows[-2], rows[-1]
			feature = {'features': tf.train.Feature(float_list = tf.train.FloatList(value = features)),
					   'label_2': tf.train.Feature(float_list = tf.train.FloatList(value = [label_2])),
					   'label_10': tf.train.Feature(float_list = tf.train.FloatList(value = [label_10]))}
			example = tf.train.Example(features = tf.train.Features(feature = feature))
			writer.write(example.SerializeToString())
def next_batch(filename, batch_size):

	len_feature = 202  #特征数（不包含标签）。 Number of features (not including tags)
	len_label = 1#标签长度。 The length of the label

	def read_data(examples):
		features = {"features": tf.io.FixedLenFeature([len_feature], tf.float32),
					"label_2": tf.io.FixedLenFeature([len_label], tf.float32),
					"label_10": tf.io.FixedLenFeature([len_label], tf.float32)}
		parsed_features = tf.io.parse_single_example(examples, features)
		return parsed_features['features'], parsed_features['label_2'], \
			   parsed_features['label_10']

	data = tf.data.TFRecordDataset(filename)
	data = data.map(read_data)
	data = data.batch(batch_size)
	print(data.take(2))
	iterator = iter(data)
	next_data, next_label_2, next_label_10 = next(iterator)

	return next_data, next_label_10, next_label_2

def make_whole_datasets(tfrecords_train, num_train_example, tfrecords_test,
						num_test_example):
	data_train, label_10_train, label_2_train = next_batch(tfrecords_train, num_train_example)
	data_test, label_10_test, label_2_test = next_batch(tfrecords_test, num_test_example)
	data, label_10, label_2 = data_train.numpy(), label_10_train.numpy(), label_2_train.numpy()
	dataset = np.concatenate([data, label_10, label_2], axis=1)

	# trainset, valiset = train_test_split(dataset, test_size = 254004,stratify=dataset['label_10'])
	trainset, valiset = train_test_split(dataset, test_size=0.125, random_state=40, stratify=dataset[:, -2])
	print("train:", trainset.shape)
	print("val:", valiset.shape)

	make_tfrecords(trainset, 'normalized/train.tfrecords')
	make_tfrecords(valiset, 'normalized/validation.tfrecords')

	del trainset, valiset


	data, label_10, label_2 = data_test.numpy(), label_10_test.numpy(), label_2_test.numpy()
	dataset = np.concatenate([data, label_10, label_2], axis=1)
	print("test:", dataset.shape)
	make_tfrecords(dataset, 'normalized/test.tfrecords')

if __name__ == '__main__':
	file_folder = 'normalized/'  # 数据标准化后存放的文件夹。A folder where data is stored after standardization
	files_train = [file_folder + str(x + 1) + '_train.tfrecords' for x in range(4)]
	files_test = [file_folder + str(x + 1) + '_test.tfrecords' for x in range(4)]
	num_train_example = 2032035  # trainset size
	num_test_example = 508012  # testset size
	make_whole_datasets(files_train, num_train_example, files_test, num_test_example)




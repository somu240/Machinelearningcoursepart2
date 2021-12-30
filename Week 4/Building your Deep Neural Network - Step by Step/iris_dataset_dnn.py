from load_data import load_dataset
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

train_data_x, test_data_x, train_output_y, test_output_y = load_dataset()
test_output_y[test_output_y == 'Iris-setosa'] = 1
test_output_y[test_output_y != 1] = 0
train_output_y[train_output_y == 'Iris-setosa'] = 1
train_output_y[train_output_y != 1] = 0
print("Number of training examples: m_train = " + str(train_data_x.shape[0]))
print("Number of testing examples: m_test = " + str(test_data_x.shape[0]))
print("train_set_x shape: " + str(train_data_x.shape))
print("train_set_y shape: " + str(train_output_y.shape))
print("test_set_x shape: " + str(test_data_x.shape))
print("test_set_y shape: " + str(test_output_y.shape))

train_data_x = train_data_x.astype(float).reshape(train_data_x.shape[0], -1).T
test_data_x = test_data_x.astype(float).reshape(test_data_x.shape[0], -1).T
train_output_y = train_output_y.astype(float).reshape(train_output_y.shape[0], -1).T
test_output_y = test_output_y.astype(float).reshape(test_output_y.shape[0], -1).T



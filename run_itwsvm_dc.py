import numpy as np
import os
from ITWSVM_DC import ITWSVM_DC
from sklearn.preprocessing import minmax_scale


def load_data(dataset_name, normalization=True):
    dataset = np.load('datasets/' + '%s.npz' % dataset_name)
    x, y = dataset['data'], dataset['label']
    if normalization:
        x = minmax_scale(x)
    return x, y


method = 'itwsvm_dc'
kernel = 'sigmoid'
dataset = 'sample_dataset'
j = -6
k = -6
w = 0

os.makedirs(os.path.join(os.path.dirname(__file__), '/save'), exist_ok=True)
save_path = os.path.dirname(__file__) + '/save/itwsvm_dc_%s' % kernel

if not os.path.exists(save_path):
    os.makedirs(save_path)

x, y = load_data(dataset, normalization=True)
print(dataset)
idx = np.load('datasets/' + dataset + '_idx.npz')
train_idx, test_idx = idx['train_idx'], idx['test_idx']
print(train_idx.shape)
# train
for i in range(10):
    x_train, x_test, y_train, y_test = x[train_idx[i]], x[test_idx[i]], y[train_idx[i]], y[test_idx[i]]
    model = ITWSVM_DC(alpha=2 ** j, gamma=2 ** k, kernel=kernel, C=2 ** w)
    model.fit(x_train, y_train)
    model.save(save_path + '/%s_%s_%d_%d_%d_%d' % (method, dataset, i, j, k, w))
    print(model.score(x_test, y_test))
# evaluation
scores = []
for i in range(10):
    x_test, y_test = x[test_idx[i]], y[test_idx[i]]
    model = ITWSVM_DC(alpha=2 ** j, gamma=2 ** k, kernel=kernel, C=2 ** w)
    model.load(save_path + '/%s_%s_%d_%d_%d_%d' % (method, dataset, i, j, k, w))
    scores.append(model.score(x_test, y_test))
print(np.mean(scores), np.std(scores))

import shutil, os, pathlib, pickle, sys, math, importlib, json
import pandas as pd
import numpy as np
from os.path import join
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.keras import layers

KERNEL_INIT_PATH = 'kernel_initializer.pk'

def rmtree(dir_path):
    print(f'Removing {dir_path}')
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)

def int_to_str(*keys):
    return [list(map(lambda elt: str(elt), key)) for key in keys]
    
def rm_mkdirp(dir_path, overwrite, quiet=False):
    if os.path.isdir(dir_path):
        if overwrite:
            if not quiet:
                print('Removing ' + dir_path)
            shutil.rmtree(dir_path, ignore_errors=True)

        else:
            print('Directory ' + dir_path + ' exists and overwrite flag not set to true.  Exiting.')
            exit(1)
    if not quiet:
        print('Creating ' + dir_path)
    pathlib.Path(dir_path).mkdir(parents=True)

def lists_to_2d_arr(list_in, max_len=None):
    '''2d list in, but where sub lists may have differing lengths, one big padded 2d arr out'''
    max_len = max([len(elt) for elt in list_in]) if max_len is None else max_len
    new_arr = np.zeros((len(list_in), max_len))
    for i,elt in enumerate(list_in):
        if len(elt) < max_len:
            new_arr[i,:len(elt)] = elt
        else:
            new_arr[i,:] = elt[:max_len]
    return new_arr

def lists_to_3d_arr(list_in):
    '''3d list in, where sub lists may have differing [1] lengths], one big padded 3d arr out'''
    max_len = max([elt.shape[0] for elt in list_in])
    new_arr = np.zeros((len(list_in), max_len, list_in[0].shape[1]))
    for i,elt in enumerate(list_in):
        new_arr[i,:len(elt)] = elt
    return new_arr

def mkdirp(dir_path):
    if not os.path.isdir(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True)

def rm_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    
def rglob(dir_path, pattern):
    return list(map(lambda elt: str(elt), pathlib.Path(dir_path).rglob(pattern)))

def move_matching_files(dir_path, pattern, new_dir, overwrite):
    rm_mkdirp(new_dir, True, overwrite)
    for elt in rglob(dir_path, pattern):
        shutil.move(elt, new_dir)
    
def subset(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.min([elt in b for elt in a]) > 0

def save_pk(file_stub, pk):
    filename = file_stub if '.pk' in file_stub else f'{file_stub}.pk'
    rm_file(filename)
    with open(filename, 'wb') as f:
        pickle.dump(pk, f)
    
def load_pk(file_stub):
    filename = file_stub if '.p' in file_stub else f'{file_stub}.pk'
    if not os.path.exists(filename):
        return {}
    
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
        return obj

def ds(d, *keys):
    '''Destructure dict. e.g.
    a = {'hey': 1, 'you': 2}
    hey, you = ds(a, 'hey', 'you')
    '''
    return [ d[k] if k in d else None for k in keys ]

def get_ints(*keys):
    return [int(key) for key in keys]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_json(file_stub, obj):
    filename = file_stub if '.json' in file_stub else f'{file_stub}.json'
    rm_file(filename)
    with open(filename, 'w') as f:
        json.dump(obj, f, cls=NumpyEncoder, indent=4)

def load_json(file_stub):
    filename = file_stub if '.json' in file_stub else f'{file_stub}.json'
    with open(filename) as json_file:
        return json.load(json_file)

def lfilter(fn, iterable):
    return list(filter(fn, iterable))

def lkeys(obj):
    return list(obj.keys())

def lvals(obj):
    return list(obj.values())

def lmap(fn, iterable):
    return list(map(fn, iterable))

def sort_dict(d, reverse=False):
    return {k: v for k,v in sorted(d.items(), key=lambda elt: elt[1], reverse=reverse)}

def dilation_pad(max_len, max_dilation_rate):
    to_ret = math.ceil(max_len/max_dilation_rate)*max_dilation_rate
    assert (to_ret % max_dilation_rate) == 0
    return to_ret

def zero_pad_to_length(data, length):
    padAm = length - data.shape[0]
    if padAm == 0:
        return data
    else:
        return np.pad(data, ((0,padAm), (0,0)), 'constant')

def paths_to_mfbs(paths, max_len):
    '''Get normalized & padded mfbs from paths'''
    # normalize
    mfbs = None
    for file_name in paths:
        if mfbs is None:
            mfbs = np.array(np.load(file_name))
        else:
            mfbs = np.concatenate([mfbs, np.load(file_name)], axis=0)
    mean_vec = np.mean(mfbs, axis=0)
    std_vec  = np.std(mfbs, axis=0)

    # concat & pad
    mfbs = None
    for file_name in paths:
        mfb = (np.load(file_name) - mean_vec) / (std_vec + np.ones_like(std_vec)*1e-3)
        if mfbs is None:
            mfbs = np.array([zero_pad_to_length(mfb, max_len)])
        else:
            mfbs = np.concatenate([mfbs, [zero_pad_to_length(mfb, max_len)]], axis=0)
    return tf.cast(mfbs, tf.float64)

def destring(y, width=3):
    ''' y is an array with elements in the format '.5;.5;0.'.  Need to turn into nx3 arr'''
    y = np.array(y)
    y_new = np.zeros((len(y), width))
    for i in range(len(y)):
        if '[0.333' in y[i]:
            y_new[i] = [.333, .333, .333]
            continue
        assert ';' in y[i] or ' ' in y[i]
        char = ';' if ';' in y[i] else None
        y_new[i] = list(map(lambda elt: float(elt), y[i].split(char)))
    return y_new

def get_batch(arr, batch_idx, batch_size):
    return arr[batch_idx * batch_size:(batch_idx + 1) * batch_size]

def sample_batch(x, y, batch_size):
    start = np.random.randint(x.shape[0]-batch_size)
    x_batch = x[start:start+batch_size]
    y_batch = y[start:start+batch_size]
    return x_batch, y_batch

def get_mfbs(paths, lengths_dict, max_dilation_rate):
    max_len = max([lengths_dict[file_name] for file_name in paths])
    max_len = dilation_pad(max_len, max_dilation_rate)
    return paths_to_mfbs(paths, max_len)

def shuffle_data(*arrs):
    rnd_state = np.random.get_state()
    for arr in arrs:
        np.random.shuffle(arr)
        np.random.set_state(rnd_state)

def get_class_weights(arr):
    '''pass in dummies'''
    class_weights = np.nansum(arr, axis=0)
    return np.sum(class_weights) / (class_weights*len(class_weights))

def get_class_weights_ds(arr):
    '''do not pass in dummies'''
    arr = np.stack(np.unique(np.array(arr), return_counts=True), axis=1)
    return (np.sum(arr[:,1]) - arr[:,1]) / arr[:,1]

def checkpt_path(model_name, iteration, models_path=MODELS_PATH, best=True):
    '''epoch==None if base'''
    model_path = join(models_path, model_name)
    mkdirp(model_path)
    if best:
        return join(model_path, f'iter{iteration}')
    else:
        return join(model_path, f'iter{iteration}_epoch{{epoch}}')

def weighted_cross_entropy(weights):
    weights = K.variable(weights)
    def loss_f(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss_f

def md_critic_loss(class_weights_ds, batch_size=BATCH_SIZE):
    def loss_f(y_true, y_pred):
        idx_x = tf.cast(tf.squeeze(tf.where(y_true[:,0]==y_true[:,0])), tf.int32)
        idx_y = tf.cast(K.argmax(y_true, axis=-1), tf.int32)
        idxs = tf.stack([idx_x, idx_y], axis=1)
        relevant_preds = tf.gather_nd(y_pred, idxs)
        weights = tf.cast(tf.gather_nd(class_weights_ds, tf.expand_dims(idx_y, 1)), tf.float64)
        modified_relevant = relevant_preds * weights
        num_ds = y_pred.shape[1]
        return (1/num_ds) * (1/batch_size) * (tf.math.reduce_sum(y_pred) - tf.math.reduce_sum(relevant_preds) - tf.math.reduce_sum(modified_relevant))
    return loss_f

def mean_sum_loss():
    def loss_f(y_true, y_pred):
        '''y_true is dummy array of dataset idxs, y_pred is softmax distribution over those idxs'''
        idx_x = tf.cast(tf.squeeze(tf.where(y_true[:,0]==y_true[:,0])), tf.int32)
        idx_y = tf.cast(K.argmax(y_true, axis=-1), tf.int32)
        idxs = tf.stack([idx_x, idx_y], axis=1)
        return tf.reduce_mean(tf.gather_nd(y_pred, idxs))
    return loss_f

def mean_sum_loss_diff():
    def loss_f(y_true, y_pred):
        idx_x_corr = tf.cast(tf.squeeze(tf.where(y_true[:,0]==y_true[:,0])), tf.int32)
        idx_y_corr = tf.cast(K.argmax(y_true, axis=-1), tf.int32)
        idxs_corr = tf.stack([idx_x_corr, idx_y_corr], axis=1)
        corr = tf.gather_nd(y_pred, idxs_corr)
        sum_corr = tf.reduce_sum(corr)
        sum_not_corr = tf.reduce_sum(y_pred) - sum_corr
        return K.max([0, sum_corr-sum_not_corr])
    return loss_f

class UAR(tf.keras.metrics.Metric):
    def __init__(self, name='uar', dtype=None):
        super(UAR, self).__init__(name=name, dtype=dtype)
        self.recall = self.add_weight(name='rec', initializer='zeros')
        self.num_updates = self.add_weight(name='nu', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_t, y_p = y_true, y_pred
        y_p = K.argmax(y_p, axis=-1)
        a = K.max(y_t, axis=-1)
        c = y_t == tf.transpose(tf.stack((a,a,a))) # map of T/F - whether elt is a max in the row
        rows = tf.cast(tf.expand_dims(tf.range(tf.shape(y_p)[0], dtype=tf.int32), 1), tf.int32)
        y_p = tf.cast(tf.expand_dims(y_p, 1), tf.int32)
        indexes = tf.concat((rows, y_p), axis=-1) # indexes corresponding to which prediction chosen at each row
        yp_updated = ((tf.cast(tf.gather_nd(c, indexes), tf.int32)) + 1) % 2
        d = tf.cast(K.argmax(y_t, axis=-1), tf.int32)
        y_p = (d + yp_updated) % 3
        y_t = d

        cm = tf.math.confusion_matrix(y_t, y_p, num_classes=3, dtype=self.dtype)
        self.recall.assign_add(tf.reduce_mean([tf.math.divide_no_nan(cm[i,i], tf.reduce_sum(cm[i,:])) for i in range(cm.shape[0])]))
        self.num_updates.assign_add(1)

    def result(self):
        return tf.math.divide_no_nan(self.recall, self.num_updates)

def get_uar(y_t, y_p):
    uar = UAR()
    uar.update_state(y_t, y_p)
    return uar.result()
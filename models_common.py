from utils import *
kernel_initializer = load_pk(KERNEL_INIT_PATH)

class GlobalMeanPoolNoZeros(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, arr):
        return K.sum(arr, axis=1) / tf.cast(tf.math.count_nonzero(arr, axis=1), tf.float64)

def num_conv_to_layer(num_conv):
    if num_conv == 2:
        return Conv_Layer2
    elif num_conv == 3:
        return Conv_Layer3
    elif num_conv == 4:
        return Conv_Layer4
    elif num_conv == 5:
        return Conv_Layer5
    else:
        assert False, f'num_conv must be in 2...5, but was {num_conv}'

class Conv_Layer2(layers.Layer):
    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation
        self.CONV1 = layers.Conv1D(filters=128, kernel_size=16, data_format='channels_last', activation='relu', padding='same', input_shape=(None, 40), use_bias=False, dilation_rate=2**(dilation*0), kernel_initializer=kernel_initializer)
        self.CONV2 = layers.Conv1D(filters=128, kernel_size=16, data_format='channels_last', activation='relu', padding='same', input_shape=(None, 40), use_bias=False, dilation_rate=2**(dilation*1), kernel_initializer=kernel_initializer)
        self.POOL = layers.MaxPool1D(pool_size=2**((dilation + 1) % 2))

    def call(self, mfbs):
        mfbs = self.CONV1(mfbs)
        mfbs = self.POOL(mfbs)
        mfbs = self.CONV2(mfbs)
        mfbs = self.POOL(mfbs)
        return mfbs

class Conv_Layer3(layers.Layer):
    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation
        self.CONV1 = layers.Conv1D(filters=128, kernel_size=8, data_format='channels_last', activation='relu', padding='same', input_shape=(None, 40), use_bias=False, dilation_rate=2**(dilation*0), kernel_initializer=kernel_initializer)
        self.CONV2 = layers.Conv1D(filters=128, kernel_size=8, data_format='channels_last', activation='relu', padding='same', input_shape=(None, 40), use_bias=False, dilation_rate=2**(dilation*1), kernel_initializer=kernel_initializer)
        self.CONV3 = layers.Conv1D(filters=128, kernel_size=8, data_format='channels_last', activation='relu', padding='same', input_shape=(None, 40), use_bias=False, dilation_rate=2**(dilation*2), kernel_initializer=kernel_initializer)
        self.POOL = layers.MaxPool1D(pool_size=2**((dilation + 1) % 2))

    def call(self, mfbs):
        mfbs = self.CONV1(mfbs)
        mfbs = self.POOL(mfbs)
        mfbs = self.CONV2(mfbs)
        mfbs = self.POOL(mfbs)
        mfbs = self.CONV3(mfbs)
        mfbs = self.POOL(mfbs)
        return mfbs

class Conv_Layer4(layers.Layer):
    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation
        self.CONV1 = layers.Conv1D(filters=128, kernel_size=4, data_format='channels_last', activation='relu', padding='same', input_shape=(None, 40), use_bias=False, dilation_rate=2**(dilation*0), kernel_initializer=kernel_initializer)
        self.CONV2 = layers.Conv1D(filters=128, kernel_size=4, data_format='channels_last', activation='relu', padding='same', input_shape=(None, 40), use_bias=False, dilation_rate=2**(dilation*1), kernel_initializer=kernel_initializer)
        self.CONV3 = layers.Conv1D(filters=128, kernel_size=4, data_format='channels_last', activation='relu', padding='same', input_shape=(None, 40), use_bias=False, dilation_rate=2**(dilation*2), kernel_initializer=kernel_initializer)
        self.CONV4 = layers.Conv1D(filters=128, kernel_size=4, data_format='channels_last', activation='relu', padding='same', input_shape=(None, 40), use_bias=False, dilation_rate=2**(dilation*3), kernel_initializer=kernel_initializer)
        self.POOL = layers.MaxPool1D(pool_size=2**((dilation + 1) % 2))

    def call(self, mfbs):
        mfbs = self.CONV1(mfbs)
        mfbs = self.POOL(mfbs)
        mfbs = self.CONV2(mfbs)
        mfbs = self.POOL(mfbs)
        mfbs = self.CONV3(mfbs)
        mfbs = self.POOL(mfbs)
        mfbs = self.CONV4(mfbs)
        mfbs = self.POOL(mfbs)
        return mfbs

class Conv_Layer5(layers.Layer):
    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation
        self.CONV1 = layers.Conv1D(filters=128, kernel_size=2, data_format='channels_last', activation='relu', padding='same', input_shape=(None, 40), use_bias=False, dilation_rate=2**(dilation*0), kernel_initializer=kernel_initializer)
        self.CONV2 = layers.Conv1D(filters=128, kernel_size=2, data_format='channels_last', activation='relu', padding='same', input_shape=(None, 40), use_bias=False, dilation_rate=2**(dilation*1), kernel_initializer=kernel_initializer)
        self.CONV3 = layers.Conv1D(filters=128, kernel_size=2, data_format='channels_last', activation='relu', padding='same', input_shape=(None, 40), use_bias=False, dilation_rate=2**(dilation*2), kernel_initializer=kernel_initializer)
        self.CONV4 = layers.Conv1D(filters=128, kernel_size=2, data_format='channels_last', activation='relu', padding='same', input_shape=(None, 40), use_bias=False, dilation_rate=2**(dilation*3), kernel_initializer=kernel_initializer)
        self.CONV5 = layers.Conv1D(filters=128, kernel_size=2, data_format='channels_last', activation='relu', padding='same', input_shape=(None, 40), use_bias=False, dilation_rate=2**(dilation*4), kernel_initializer=kernel_initializer)
        self.POOL = layers.MaxPool1D(pool_size=2**((dilation + 1) % 2))

    def call(self, mfbs):
        mfbs = self.CONV1(mfbs)
        mfbs = self.POOL(mfbs)
        mfbs = self.CONV2(mfbs)
        mfbs = self.POOL(mfbs)
        mfbs = self.CONV3(mfbs)
        mfbs = self.POOL(mfbs)
        mfbs = self.CONV4(mfbs)
        mfbs = self.POOL(mfbs)
        mfbs = self.CONV5(mfbs)
        mfbs = self.POOL(mfbs)
        return mfbs

class Conv_Pool_Layer(layers.Layer):
    def __init__(self, conv_layer):
        super().__init__()
        self.conv_layer = conv_layer
        self.POOL = layers.GlobalMaxPool1D(data_format='channels_last')
        self.DROP = layers.Dropout(rate=0.2)

    def call(self, mfbs):
        mfbs = self.conv_layer(mfbs)
        mfbs = self.POOL(mfbs)
        mfbs = self.DROP(mfbs)
        return mfbs

class CLF_Layer(layers.Layer):
    def __init__(self, num_out=3):
        super().__init__()
        self.DENSE1 = layers.Dense(128, activation='relu', kernel_initializer=kernel_initializer)
        self.DENSE2 = layers.Dense(128, activation='relu', kernel_initializer=kernel_initializer)
        self.DENSE3 = layers.Dense(num_out, activation='softmax', kernel_initializer=kernel_initializer)

    def call(self, mfbs):
        mfbs = self.DENSE1(mfbs)
        mfbs = self.DENSE2(mfbs)
        mfbs = self.DENSE3(mfbs)
        return mfbs

class Sigmoid_CLF(layers.Layer):
    def __init__(self):
        super().__init__()
        self.DENSE1 = layers.Dense(128, activation='linear', kernel_initializer=kernel_initializer)
        self.DENSE2 = layers.Dense(128, activation='linear', kernel_initializer=kernel_initializer)
        self.DENSE3 = layers.Dense(1, activation='sigmoid', kernel_initializer=kernel_initializer)

    def call(self, mfbs):
        mfbs = self.DENSE1(mfbs)
        mfbs = self.DENSE2(mfbs)
        mfbs = self.DENSE3(mfbs)
        return mfbs

class Linear_CLF(layers.Layer):
    def __init__(self):
        super().__init__()
        self.DENSE1 = layers.Dense(128, activation='linear', kernel_initializer=kernel_initializer)
        self.DENSE2 = layers.Dense(128, activation='linear', kernel_initializer=kernel_initializer)
        self.DENSE3 = layers.Dense(1, activation='linear', kernel_initializer=kernel_initializer)

    def call(self, mfbs):
        mfbs = self.DENSE1(mfbs)
        mfbs = self.DENSE2(mfbs)
        mfbs = self.DENSE3(mfbs)
        return mfbs

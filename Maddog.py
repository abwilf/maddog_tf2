from metrics import *
from models_common import *
from utils import *

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
MD_CRITIC_MIN_WEIGHT = -.01
MD_CRITIC_MAX_WEIGHT = .01
md_weight_clip = lambda w: tf.clip_by_value(w, MD_CRITIC_MIN_WEIGHT, MD_CRITIC_MAX_WEIGHT)
kernel_initializer = load_pk(KERNEL_INIT_PATH)
tf.keras.backend.set_floatx('float64')

class MD_Critic_Layer(layers.Layer):
    def __init__(self, num_datasets, base_type):
        super().__init__()
        self.DENSE1 = layers.Dense(128, activation='linear', kernel_initializer=kernel_initializer, kernel_constraint=md_weight_clip)
        self.DENSE2 = layers.Dense(128, activation='linear', kernel_initializer=kernel_initializer, kernel_constraint=md_weight_clip)
        self.DENSE3 = layers.Dense(num_datasets, activation='linear', kernel_initializer=kernel_initializer, kernel_constraint=md_weight_clip)

    def call(self, mfbs):
        mfbs = self.DENSE1(mfbs)
        mfbs = self.DENSE2(mfbs)
        mfbs = self.DENSE3(mfbs)
        return mfbs

class Critic_Model(tf.keras.Model):
    def __init__(self, conv_pool_layer, critic_layer):
        super().__init__()
        self.conv_pool_layer = conv_pool_layer
        self.critic_layer = critic_layer

    def call(self, mfbs):
        mfbs = self.conv_pool_layer(mfbs)
        mfbs = self.critic_layer(mfbs)
        return mfbs

class Maddog(tf.keras.Model):
    def __init__(self, num_conv, dilation, critic_layer, batch_size, ltar_zero, num_datasets):
        super().__init__()
        conv_layer = num_conv_to_layer(num_conv)(dilation)
        self.conv_pool_layer = Conv_Pool_Layer(conv_layer)
        self.clf_layer = CLF_Layer()
        self.critic_layer = critic_layer

        self.conv_pool_layer.trainable=True
        self.clf_layer.trainable=True
        self.critic_layer.trainable=False

        self.ones_placeholder = tf.ones((2,num_datasets), dtype=tf.float64)
        self.ltar_zero = ltar_zero

    def call(self, x, training=False):
        if training:
            src_mfbs, tar_mfbs, ltar_mfbs = x['src'], x['tar'], x['ltar']
            src_enc = self.conv_pool_layer(src_mfbs)
            tar_enc = self.conv_pool_layer(tar_mfbs)

            src_emo = self.clf_layer(src_enc)
            if not self.ltar_zero:
                ltar_enc = self.conv_pool_layer(ltar_mfbs)
                ltar_emo = self.clf_layer(ltar_enc)
                emo = tf.concat([src_emo, ltar_emo], axis=0)
            else:
                emo = src_emo

            src_ds = self.critic_layer(src_enc)
            tar_ds = self.critic_layer(tar_enc)
            ds = tf.concat([src_ds, tar_ds], axis=0)

            return emo, ds

        else:
            mfbs = x['test']
            mfbs = self.conv_pool_layer(mfbs)
            emo = self.clf_layer(mfbs)
            return emo, self.ones_placeholder

def get_gen_md_critic(df, max_dilation_rate, base_type, batch_size=BATCH_SIZE, num_datasets=None):
    df = df[['features', 'dataset']].to_numpy()
    x, d = df[:, 0], df[:, 1]
    d = pd.get_dummies(d).to_numpy().astype('float64')
    return Maddog_Critic_Generator(x, d, batch_size, max_dilation_rate)

class Maddog_Critic_Generator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size, max_dilation_rate, lengths_path=f'{BASE_DIR}/csvs/lengths.csv'):
        '''y: dataset idxs'''
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.max_dilation_rate = max_dilation_rate
        self.lengths_dict = {row['features']: row['length'] for _, row in pd.read_csv(lengths_path).iterrows()}

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, batch_idx):
        batch_x, batch_y = sample_batch(self.x, self.y, self.batch_size)
        max_len = max([self.lengths_dict[file_name] for file_name in batch_x])
        max_len = dilation_pad(max_len, self.max_dilation_rate)
        batch_x = paths_to_mfbs(batch_x, max_len)
        
        return batch_x, batch_y

def get_gen_md(args, batch_size=BATCH_SIZE, training=True, testing=False):
    df_src, df_tar, df_val, df_ltar, max_dilation_rate, base_type, df_test = ds(args, 'df_src', 'df_tar', 'df_val', 'df_ltar', 'max_dilation_rate', 'base_type', 'df_test')
    if training:
        x_src, x_tar, x_ltar = df_src['features'].to_numpy(), df_tar['features'].to_numpy(), df_ltar['features'].to_numpy()
        
        y_src_emo = destring(df_src['emotion']).astype('float64')
        y_ltar_emo = destring(df_ltar['emotion']).astype('float64')
        
        y_src_ds = df_src['dataset'].to_numpy()
        y_tar_ds = df_tar['dataset'].to_numpy()

        x = [x_src, x_tar, x_ltar]
        y = [y_src_emo, y_ltar_emo, y_src_ds, y_tar_ds]

        noisy = args['noisy_train']
    
    else:
        x = df_val['features'].to_numpy()
        y = destring(df_val['emotion']).astype('float64')
        noisy = 0

    if testing:
        x = df_test['features'].to_numpy()
        y = destring(df_test['emotion']).astype('float64')
        noisy = 0
    return Maddog_Generator(x, y, noisy, batch_size, max_dilation_rate, args['num_datasets'], training=training)

class Maddog_Generator(keras.utils.Sequence):
    def __init__(self, x, y, noisy, batch_size, max_dilation_rate, num_datasets, training=True, lengths_path=f'{BASE_DIR}/csvs/lengths.csv'):
        super().__init__()
        if training:
            self.x_src, self.x_tar, self.x_ltar = x
            self.y_src_emo, self.y_ltar_emo, self.y_src_ds, self.y_tar_ds = y
        else:
            self.x = x
            self.y = y

        self.num_datasets = num_datasets
        self.training = training
        self.batch_size = batch_size
        self.max_dilation_rate = max_dilation_rate
        self.lengths_dict = {row['features']: row['length'] for _, row in pd.read_csv(lengths_path).iterrows()}

        if noisy:
            f = lambda elt: elt.replace('orig', 'noisy')
            self.x_src = np.array(list(map(f, self.x_src)))

        self.noisy = noisy

    def __len__(self):
        x = self.x_src if self.training else self.x
        return math.ceil(x.shape[0] / self.batch_size)

    def __getitem__(self, batch_idx):
        if self.training:
            # GET batch from src
            x_src_paths = get_batch(self.x_src, batch_idx, self.batch_size)
            x_src = get_mfbs(x_src_paths, self.lengths_dict, self.max_dilation_rate)
        
            y_src_emo = get_batch(self.y_src_emo, batch_idx, self.batch_size)
            y_src_ds = get_batch(self.y_src_ds, batch_idx, self.batch_size)

            # SAMPLE batches from tar, ltar
            x_tar_paths, y_tar_ds = sample_batch(self.x_tar, self.y_tar_ds, self.batch_size)
            x_tar = get_mfbs(x_tar_paths, self.lengths_dict, self.max_dilation_rate)

            if len(self.x_ltar) == 0:
                x_ltar = np.zeros(1)
                y_ltar_emo = np.zeros((0, y_src_emo.shape[1]))
            else:
                x_ltar_paths, y_ltar_emo = sample_batch(self.x_ltar, self.y_ltar_emo, self.batch_size)
                x_ltar = get_mfbs(x_ltar_paths, self.lengths_dict, self.max_dilation_rate)

            x = {'src': x_src, 'tar': x_tar, 'ltar': x_ltar, 'test': x_src} # only pass 'test' bc tf graph needs consistency with validation
            
            y_emo = np.concatenate([y_src_emo, y_ltar_emo], axis=0)
            y_ds = np.concatenate([y_src_ds, y_tar_ds], axis=0)

            y_ds = pd.get_dummies(y_ds).to_numpy().astype('float64')

            y = y_emo, y_ds

        else:
            x_paths = get_batch(self.x, batch_idx, self.batch_size)
            x = get_mfbs(x_paths, self.lengths_dict, self.max_dilation_rate)
            x = {'src': x, 'tar': x, 'ltar': x, 'test': x} # have to include all elts b/c of tf graph
            y1 = get_batch(self.y, batch_idx, self.batch_size)
            y = tf.constant(y1), tf.ones((2,self.num_datasets), dtype=tf.float64)
            
        return x, y

    def on_epoch_end(self):
        if self.training:
            shuffle_data(self.x_src, self.y_src_emo, self.y_src_ds)

class Critic_Callback(tf.keras.callbacks.Callback):
    def __init__(self, critic_model, critic_gen, num_steps_critic):
        super().__init__()
        self.critic_gen = critic_gen
        self.num_steps_critic = num_steps_critic
        self.critic_model = critic_model

    def on_epoch_begin(self, epoch, logs=None):
        print(f'Training critic...')
        self.critic_model.conv_pool_layer.trainable = False
        self.critic_model.critic_layer.trainable = True
        self.critic_model.conv_pool_layer.set_weights(self.model.conv_pool_layer.get_weights())
        self.critic_model.fit(
            x=self.critic_gen,
            epochs=1,
            steps_per_epoch=self.num_steps_critic,
            verbose=1,
        )
        self.model.critic_layer.set_weights(self.critic_model.critic_layer.get_weights())
        self.model.critic_layer.trainable = False
        self.model.conv_pool_layer.trainable = True
        print(f'Training full...')


class Maddog_Wrapper():
    def __init__(self, args):
        self.args = args
        self.critic_gen = get_gen_md_critic(args['df_tot'], max_dilation_rate=args['max_dilation_rate'], base_type=args['base_type'], num_datasets=args['num_datasets'])
        self.train_gen = get_gen_md(args, batch_size=args['batch_size'], training=True)
        self.val_gen = get_gen_md(args, batch_size=args['batch_size'], training=False)
        
        # clf loss
        class_arr = destring(args['df_train']['emotion'].to_numpy())
        class_weights_emotion = get_class_weights(class_arr)
        self.clf_loss = weighted_cross_entropy(class_weights_emotion)

        # critic loss
        class_arr = args['df_tot']['dataset'].to_numpy()
        class_weights_ds = get_class_weights_ds(class_arr)
        self.critic_loss = md_critic_loss(class_weights_ds)
        self.critic_loss_inverse = mean_sum_loss()
        self.args['num_steps_critic'] = 5
        md_lambda = 1e-1


        # set up models
        critic_layer = MD_Critic_Layer(args['num_datasets'], args['base_type'])
        self.full_model = Maddog(args['num_conv'], args['dilation'], critic_layer, args['batch_size'], ltar_zero=args['amt_tar']==0, num_datasets=args['num_datasets'])
        self.full_model.compile(
            tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss={'output_1': self.clf_loss, 'output_2': self.critic_loss_inverse},
            loss_weights={'output_1': 1, 'output_2': md_lambda},
            metrics={'output_1': [UAR()]},
        )

        self.critic_model = Critic_Model(self.full_model.conv_pool_layer, critic_layer)
        self.critic_model.compile(
            tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=self.critic_loss,
        )
    
    def fit(self):
        history = self.full_model.fit(
            x=self.train_gen,
            validation_data=self.val_gen,
            epochs=self.args['num_epochs'],
            steps_per_epoch=self.args['steps_per_epoch'], # for testing code - usually this is None
            validation_steps=self.args['validation_steps'],
            verbose=1,
            use_multiprocessing=True,
            workers=4,
            callbacks=[
                Critic_Callback(self.critic_model, self.critic_gen, self.args['num_steps_critic']),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpt_path(self.args['model_name'], iteration=self.args['iteration']),
                    save_best_only=True, 
                    save_weights_only=True, 
                    monitor='val_output_1_uar', 
                    mode='max',
                    save_format='tf'
                )
            ]
        )
        
        return history.history['val_output_1_uar'], self.full_model


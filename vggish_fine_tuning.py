import vggish_slim
import vggish_input
from tensorflow.keras.models import Sequential

import pickle
import tensorflow.compat.v1 as tf1
import numpy as np
import tensorflow.keras.layers as lyr
import tensorflow as tf
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def extract_feature_train_vggish(df):
    for _,row in df.iterrows():
        filepath= row['id']
        examples = vggish_input.wavfile_to_examples(filepath)
        label = row['encoded_genres']
        for i in range(examples.shape[0]):
            example_dim =np.expand_dims(examples[i], axis=-1)
            features_tensor = tf.convert_to_tensor(example_dim, dtype=tf.float32)
            #label_tensor = tf.convert_to_tensor(label, dtype=tf.int32)
            yield features_tensor, label



def extract_weight_vggish(checkpoint_path):
    weights_save_path = 'model_weights.pkl'
    with tf1.Graph().as_default(), tf1.Session() as session:
        vggish_slim.define_vggish_slim(training=False)
        vggish_var_names = [v.name for v in tf1.global_variables()]
        vggish_vars = [v for v in tf1.global_variables() if v.name in vggish_var_names]

        saver = tf1.train.Saver(vggish_vars, name='vggish_load_pretrained', write_version=1)
        saver.restore(session, checkpoint_path)

        model_wts = {}
        for var in vggish_vars:
            try:
                model_wts[var.name] = var.eval()
            except:
                print("For var={}, an exception occurred".format(var.name))
    with open(weights_save_path, 'wb') as f:
        pickle.dump(model_wts, f)
    return model_wts

def load_vggish_model(model_wts,trainble):
    vggish = Sequential()
    
    input_layer = lyr.Input(shape=(96, 64, 1), name='input_layer')
    vggish.add(input_layer)
    ######################################
    ############# CONV LAYER ############# 
    ######################################
    #no weights for pooling layers
    conv1  = lyr.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same', activation='relu', name='conv1')
    vggish.add(conv1)
    conv1.set_weights([model_wts['vggish/conv1/weights:0'], model_wts['vggish/conv1/biases:0']])
    conv1.trainable = trainble

    pool1  = lyr.MaxPooling2D(pool_size=(2,2), strides=2, name='pool1')
    vggish.add(pool1)

    conv2  = lyr.Conv2D(filters=128, kernel_size=(3,3), strides=1, padding='same', activation='relu', name='conv2')
    vggish.add(conv2)
    conv2.set_weights([model_wts['vggish/conv2/weights:0'], model_wts['vggish/conv2/biases:0']])
    conv2.trainable = trainble

    pool2  = lyr.MaxPooling2D(pool_size=(2,2), strides=2, name='pool2')
    vggish.add(pool2)

    conv3_1= lyr.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation='relu', name='conv3_1')
    vggish.add(conv3_1)
    conv3_1.set_weights([model_wts['vggish/conv3/conv3_1/weights:0'], model_wts['vggish/conv3/conv3_1/biases:0']])
    conv3_1.trainable = trainble

    conv3_2= lyr.Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same', activation='relu', name='conv3_2')
    vggish.add(conv3_2)
    conv3_2.set_weights([model_wts['vggish/conv3/conv3_2/weights:0'], model_wts['vggish/conv3/conv3_2/biases:0']])
    conv3_2.trainable = trainble

    pool3  = lyr.MaxPooling2D(pool_size=(2,2), strides=2, name='pool3')
    vggish.add(pool3)

    conv4_1= lyr.Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', activation='relu', name='conv4_1')
    vggish.add(conv4_1)
    conv4_1.set_weights([model_wts['vggish/conv4/conv4_1/weights:0'], model_wts['vggish/conv4/conv4_1/biases:0']])
    conv4_1.trainable = trainble

    conv4_2= lyr.Conv2D(filters=512, kernel_size=(3,3), strides=1, padding='same', activation='relu', name='conv4_2')
    vggish.add(conv4_2)
    conv4_2.set_weights([model_wts['vggish/conv4/conv4_2/weights:0'], model_wts['vggish/conv4/conv4_2/biases:0']])
    conv4_2.trainable = trainble

    pool4  = lyr.MaxPooling2D(pool_size=(2,2), strides=2, name='pool4')
    vggish.add(pool4)

    ######################################
    #############NN-FC LAYER ############# 
    ######################################

    vggish.add(lyr.Flatten())

    fc1_1  = lyr.Dense(4096, activation='relu', name='fc1_1')
    vggish.add(fc1_1)
    fc1_1.set_weights([model_wts['vggish/fc1/fc1_1/weights:0'], model_wts['vggish/fc1/fc1_1/biases:0']])
    fc1_1.trainable = trainble

    fc1_2  = lyr.Dense(4096, activation='relu', name='fc1_2')
    vggish.add(fc1_2)
    fc1_2.set_weights([model_wts['vggish/fc1/fc1_2/weights:0'], model_wts['vggish/fc1/fc1_2/biases:0']])
    fc1_2.trainable = trainble

    fc2  = lyr.Dense(128, activation='relu', name='fc2')
    vggish.add(fc2)
    fc2.set_weights([model_wts['vggish/fc2/weights:0'], model_wts['vggish/fc2/biases:0']])
    fc2.trainable = trainble

    fc_last = lyr.Dense(10,activation='softmax')
    vggish.add(fc_last)

    return vggish

def load_embeddings_from_df(df):
    def gen_funct(df):
        for _,row in df.iterrows():
            filepath= row['id']
            examples = vggish_input.wavfile_to_examples(filepath)
            embedding = model.predict(examples)
            tf_embedding = tf.convert_to_tensor(embedding,dtype=tf.float32)
            label_tensor = tf.convert_to_tensor(label, dtype=tf.int32)
            yield tf_embedding,label_tensor

    

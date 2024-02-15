import pickle
import pandas as pd
import os
import ast
import numpy as np
import sys
import librosa
sys.path.append("\vggish")


#from tensorflow.keras.optimizers import scheduler
import tensorflow as tf
import tensorflow.keras.layers as lyr
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall,CategoricalAccuracy
#from keras.models import Model,Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import preprocessing as pp
import vggish_fine_tuning as vft
import vggish_input
from keras.callbacks import Callback
import seaborn as sns
from sklearn.metrics import silhouette_score
data = []
def get_k_means_df(df):
    for _,row in df.iterrows():
        y,sr = librosa.load(row['id'])

        tempogram=librosa.feature.tempogram(y=y,sr=sr, hop_length = 512)
        tempogram_mean=np.mean(tempogram,axis=1) 

        chroma = librosa.feature.chroma_stft(y=y,sr=sr)
        chroma_mean=np.mean(chroma, axis=1)


        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfcc_mean=np.mean(mfcc,axis=1)

        sc = librosa.feature.spectral_centroid(y=y,sr=sr) 
        sc_mean=np.mean(sc) 

        sr=librosa.feature.spectral_rolloff(y=y,sr=sr)
        sr_mean=np.mean(sr)
        row_dict = {
            **{f'chroma{i+1}_mean': chroma_mean[i] for i in range(chroma_mean.shape[0])},
            **{f'tempogram{i+1}_mean' : tempogram_mean[i] for i in range(tempogram_mean.shape[0])},
            **{f'mfcc{i+1}_mean' : mfcc_mean[i] for i in range(mfcc_mean.shape[0])},
            'sc' : sc_mean,
            'sr' : sr_mean
            }
        data.append(row_dict) 
    df_mean = pd.Dataframe(data)
    #df_mean.to_csv("k-mean_labels")
    max_clusters = 20
    inertia_values = []
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df_mean)
        inertia_values.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df_mean, kmeans.labels_))

    # Plot elbow
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), inertia_values, marker='o')
    plt.title('Elbow Method')
    plt.xlabel(' K')
    plt.ylabel('Inertia')

    # Plot del silhouette coefficient
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette coefficient')
    plt.xlabel('K')
    plt.ylabel('Silhouette Score')

    plt.tight_layout()
    plt.show()

    #optimal k with elbow method
    optimal_k = np.argmin(inertia_values) + 2  # +2 perché inizia da K=2

    # K means to optimal_k
    kmeans_optimal = KMeans(n_clusters=optimal_k)
    kmeans_optimal.fit(df_mean)

    # PCA to visualize cluster in 2 dimension
    pca = PCA(n_components=3)
    df_pca = pca.fit_transform(df_mean)

    # Plotta i risultati del K-Means applicato al K migliore dopo PCA
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    #Plot color based
    scatter = ax.scatter(df_pca[:, 0], df_pca[:, 1], df_pca[:, 2], c=kmeans_optimal.labels_, cmap='viridis', alpha=0.8)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    legend = ax.legend(*scatter.legend_elements(), title='Cluster')
    ax.add_artist(legend)
    plt.title('K-Means after PCA in 3D')

    plt.show()
    return df_mean

from sklearn.preprocessing import StandardScaler
df=pd.read_csv("k-mean_labels.csv")
scaler = StandardScaler()
df = scaler.fit_transform(df)




############################################
############## DATABASE LOAD ###############
############################################


id_genres = pd.read_csv("..\Dataset\id_genres.csv", delimiter='\t')
#id_information = pd.read_csv("..\Dataset\id_information.csv", delimiter='\t', index_col='id')
id_metadata = pd.read_csv("..\Dataset\id_metadata.csv", delimiter='\t')
#id_tags = pd.read_csv("..\Dataset\id_tags.csv", delimiter='\t', index_col='id')

#######################SUBGENRE FOR PRE PROCESSING#######################
main_genres = ['hip hop', 'rock', 'jazz','folk','reggae','electronic','pop','country', 'punk','metal','rap']
nationality= ['spanish','chinese','nigerian','french','latin','polish','dutch','uk','german','irish','en espanol','italian','australian','mexican','russian',
              'turkish','brazilian','portuguese','belgian','chilean','armenian','brithish','scottish','greek','chileno','lithuanian','indonesian','romanian',
              'cumbia','taiwan','swedish','armenian','colombian','bulgarian','argentino','albanian','polish','kazakh','thai','finnish','norwegian','chileno',
              'italo','canadian']
subgenre_remove = ['christian','skinhead reggae','rock nacional','crack rock steady','yacht rock','pinoy rock','lovers rock','reggae fusion','meme rap',
                   'beach house','tribal house','dub metal','cyber metal','sleaze rock','disney']
subgenre_divide = ['folk rock', 'rap rock','jazz rap','jazz funk','jazz metal','reggae rock','electronic rock','country pop','pop folk','pop rock','pop rap',
                   'country rap','country rock','pop punk','rap metal']

proc_df = pd.merge(id_genres, id_metadata[['id','popularity']], left_on='id', right_on='id', how='inner')
pp_df=pp.preprocess(proc_df,nationality,subgenre_remove,subgenre_divide,main_genres,10)

if 'hip hop' in main_genres:
    main_genres.remove('hip hop')

pp_df.to_csv('preprocessed_labels.csv')


pp_df['genres_count'] = pp_df['genres'].str.split(',').apply(len)
df = pp_df[pp_df['genres_count'] <= 2]
df_1 = pp_df[pp_df['genres_count']==1]
pp_df=pd.DataFrame()
for genre in main_genres:
    df_genre = df_1[df_1['genres'].str.contains(genre)]
    pp_df=pd.concat([pp_df,df_genre.sample(650)])

pp_df.drop_duplicates()


pp_df['genres'] = pp_df['genres'].apply(lambda x: ast.literal_eval(x))
genre_dict = {
    'rock': 0,
    'jazz': 1,
    'folk': 2,
    'reggae': 3,
    'electronic': 4,
    'pop': 5,
    'country': 6,
    'punk': 7,
    'metal': 8,
    'rap': 9
}
###
pp_df['encoded_genres'] = pp_df['genres'].apply(lambda x: np.eye(10)[genre_dict[x[0]]])
# Creazione della colonna contenente il vettore binario
pp_df['encoded_genres'] = pp_df[main_genres2].values.tolist()
pp_df['id'] = pp_df['id'].apply(lambda x: f"C:\\Users\\39371\\Desktop\\MusicGenreClassifier\\Dataset\\audios\\{x}.mp3")
pp_df['id'] = pp_df['id'].astype(str)
pp_df=pp_df[['id','encoded_genres']]
pp_df=pp_df.sample(frac=1,random_state=42)
pp_df=pp_df.reset_index(drop=True)

pp_df.to_csv('singlelabel_dataset.csv')

########################################
########## VGGISH FINE TUNING ##########
########################################


train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)
def extract_feature_train_vggish(df):
    for _,row in df.iterrows():
        filepath= row['id']
        examples = vggish_input.wavfile_to_examples(filepath)
        label = np.fromstring(row['encoded_genres'][1:-1], sep=' ')
        for i in range(examples.shape[0]):
            example_dim =np.expand_dims(examples[i], axis=-1)
            features_tensor = tf.convert_to_tensor(example_dim, dtype=tf.float32)
            label_tensor = tf.convert_to_tensor(label, dtype=tf.int32)
            yield features_tensor, label_tensor
train = tf.data.Dataset.from_generator(
                                    lambda: extract_feature_train_vggish(train_df),
                                    output_signature = (
                                        tf.TensorSpec(shape=(96, 64,1), dtype=tf.float32),
                                        tf.TensorSpec(shape=(10,), dtype=tf.int32)
                                    )
)
train.save(path=r"C:\Users\39371\Desktop\MusicGenreClassifier\MusicGenreClassifier\Dataset\TrainSL")
test = tf.data.Dataset.from_generator(
                                    lambda: extract_feature_train_vggish(test_df),
                                    output_signature = (
                                        tf.TensorSpec(shape=(96, 64,1), dtype=tf.float32),
                                        tf.TensorSpec(shape=(10,), dtype=tf.int32)
                                    )
)
test.save(path=r"C:\Users\39371\Desktop\MusicGenreClassifier\MusicGenreClassifier\Dataset\TestSL")


#train = tf.data.Dataset.load(path=r"C:\Users\39371\Desktop\MusicGenreClassifier\MusicGenreClassifier\Dataset\TrainSL")
#test = tf.data.Dataset.load(path=r"C:\Users\39371\Desktop\MusicGenreClassifier\MusicGenreClassifier\Dataset\TestSL")
train_dataset = train.shuffle(4000).batch(64).prefetch(tf.data.AUTOTUNE)
test_dataset = test.shuffle(1500).batch(64).prefetch(tf.data.AUTOTUNE)

with open('model_weights.pkl', "rb") as file:
    model_wts = pickle.load(file)
vggish_model = vft.load_vggish_model(model_wts,trainble=True)



class PerClassMetrics(Callback):
    def __init__(self, test):
        super().__init__()
        self.test = test
    def on_epoch_end(self, epoch, logs=None):
        log_mels = self.test.batch(32).map(lambda x, _: x)
        true_predictions = self.test.map(lambda _, y: y)
        true_predictions_list = [y.numpy() for y in true_predictions]
        true_predictions_np = np.array(true_predictions_list)
        print(true_predictions_np.shape)
        predictions = self.model.predict(log_mels)
        true_classes = np.argmax(true_predictions_np, axis=1)
        predicted_classes = np.argmax(predictions, axis=1)
        #predictions = (predictions >= 0.65).astype(int)
        original_class_names = ['rock', 'jazz', 'folk', 'reggae', 'electronic', 'pop', 'country', 'punk', 'metal', 'rap']
        class_report = classification_report(true_classes,predicted_classes,target_names=original_class_names)
        print(class_report)

# Creare un'istanza della callback
mia_callback = PerClassMetrics(test)
vggish_model.compile(optimizer=Adam(learning_rate=0.000004), loss='categorical_crossentropy', metrics=['accuracy',Precision(),Recall()])
model_checkpoint = ModelCheckpoint('vggish_best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True, verbose=1)
history = vggish_model.fit(train_dataset, epochs=15, validation_data=test_dataset,callbacks=[model_checkpoint, early_stopping,mia_callback],verbose=1)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


########################################
############# RNN TRAINING #############
########################################
#vggish_model = load_model('vggish_best_model.h5')

vggish_model.pop() #remove last layer for embeddings
df = pd.read_csv('singlelabel_dataset.csv')
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)
def gen_funct(df):
        for _,row in df.iterrows():
            filepath= row['id']
            examples = vggish_input.wavfile_to_examples(filepath)
            examples = np.expand_dims(examples, axis=-1)
            embedding = vggish_model.predict(examples)
            print(embedding.shape)
            label = np.fromstring(row['encoded_genres'][1:-1], sep=' ')
            tf_embedding = tf.convert_to_tensor(embedding,dtype=tf.float32)
            label_tensor = tf.convert_to_tensor(label, dtype=tf.int32)
            yield tf_embedding,label_tensor
train1 = tf.data.Dataset.from_generator(
                                    lambda: gen_funct(train_df),
                                    output_signature = (
                                        tf.TensorSpec(shape=(None,128), dtype=tf.float32),
                                        tf.TensorSpec(shape=(10,), dtype=tf.int32)
                                    )
)
train1.save(path=r"C:\Users\39371\Desktop\MusicGenreClassifier\MusicGenreClassifier\Dataset\TrainEMB")
test1 = tf.data.Dataset.from_generator(
                                    lambda: gen_funct(test_df),
                                    output_signature = (
                                        tf.TensorSpec(shape=(None,128), dtype=tf.float32),
                                        tf.TensorSpec(shape=(10,), dtype=tf.int32)
                                    )
)
test1.save(path=r"C:\Users\39371\Desktop\MusicGenreClassifier\MusicGenreClassifier\Dataset\TestEMB")


#train1=tf.data.Dataset.load(path=r"C:\Users\39371\Desktop\MusicGenreClassifier\MusicGenreClassifier\Dataset\TrainEMB")
#test1=tf.data.Dataset.load(path=r"C:\Users\39371\Desktop\MusicGenreClassifier\MusicGenreClassifier\Dataset\TestEMB")

train_dataset = train1.shuffle(2000).batch(16).prefetch(tf.data.AUTOTUNE)
test_dataset = test1.shuffle(600).batch(16).prefetch(tf.data.AUTOTUNE)
from tensorflow.keras.layers import Bidirectional, LSTM, Dense,BatchNormalization
model_checkpoint = ModelCheckpoint('RNN_model.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min', restore_best_weights=True, verbose=1)
rnn_model_bidirectional = Sequential([
    Bidirectional(LSTM(units=512, return_sequences=True), input_shape=(31, 128)),
    Bidirectional(LSTM(units=256)),
    Dense(units=256, activation='relu'),
    Dense(units=10, activation='sigmoid') 
])
rnn_model_bidirectional.compile(optimizer=Adam(learning_rate=0.000002), loss='categorical_crossentropy', metrics=['accuracy',Precision(),Recall()])
history=rnn_model_bidirectional.fit(train_dataset, epochs=20, validation_data=test_dataset,callbacks=[model_checkpoint, early_stopping,mia_callback],verbose=1)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Stampa l'andamento della accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


'''
df = read_json('yelp_academic_dataset_review.json')
df1= pd.read_csv('example.csv')
df = df1.sample(frac=0.1)
df['text'] = df['text'].apply(preprocess_text)
df['num_words'] = df['text'].apply(lambda x: len(x.split()))
df.to_csv('preprocess.csv')
print(df['num_words'].max())
#REMOVE STOPWORD AND LOWER CASE
plt.hist(df['num_words'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Numero di parole')
plt.ylabel('Frequenza')
plt.title('Distribuzione del numero di parole')
plt.show()
word_counts = df['num_words'].value_counts()
print(word_counts)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", down_lower_case=True)

input_ids = [tokenizer.encode(sent, add_special_tokens=True, max_length=156, pad_to_max_length=True, truncation=True) for sent in df['text']]#####

attention_masks = [[float(i>0) for i in seq] for seq in input_ids]
from sklearn.preprocessing import OneHotEncoder
Y1 = df['stars'].values#####
Y1_shifted = Y1 - 1
encoder = OneHotEncoder(categories='auto', sparse=False)
Y = encoder.fit_transform(Y1_shifted.reshape(-1, 1))
print('finito')

train_inputs, validation_inputs, train_masks, validation_masks, train_labels, validation_labels = train_test_split(input_ids,attention_masks, Y, random_state=42, test_size=0.2)

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
validation_labels = torch.tensor(validation_labels, dtype=torch.float32)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)
#3)DataLoader for training and test dataset to iterate over batches of data
#batches dimension = 32 to not overload GPU
batch_size = 16
train_data = TensorDataset(train_inputs,train_masks,train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data,sampler=train_sampler,batch_size=batch_size,pin_memory=True)

validation_data = TensorDataset(validation_inputs,validation_masks,validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data,sampler=validation_sampler,batch_size=batch_size,pin_memory=True)
#hyper parameter : :
lr= 2e-5
adam_epsilon = 1e-8
epochs=15
num_warmup_steps=0

num_training_steps = len(train_dataloader)*epochs
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

optimizer = AdamW(model.parameters(), lr=lr,eps=adam_epsilon,correct_bias=False) 
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
model.zero_grad()
loss_function = torch.nn.CrossEntropyLoss()

#CUDA for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)
SEED = 19
np.random.seed(SEED)
torch.manual_seed(SEED)
if device == torch.device("cuda"):
    torch.cuda.manual_seed_all(SEED)
model.to(device)

train_loss_set = []
learning_rate = []
#training phase

print('inizia il training')
model.train()  # set model in training phase

for epoch in range(epochs):
    label_predictions = {}
    batch_loss = 0
    conta=0
    for batch_inputs, batch_masks, batch_labels in train_dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_masks = batch_masks.to(device)
        batch_labels = batch_labels.to(device)
        print('conta')
        print(conta)
        conta = conta + 1
        # forward pass
        outputs = model(batch_inputs, attention_mask=batch_masks, labels=batch_labels)
        logits = outputs.logits
        # calculate loss
        loss = loss_function(logits, batch_labels.float())

        # gradient and weight update
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        optimizer.zero_grad()
        batch_loss += loss.item()

    avg_train_loss = batch_loss / len(train_dataloader)
     # store the current learning rate
    for param_group in optimizer.param_groups:
        print("\n\tCurrent Learning rate: ", param_group['lr'])
        learning_rate.append(param_group['lr'])

    train_loss_set.append(avg_train_loss)
    print(F'\n\tAverage Training loss: {avg_train_loss}')
    predictions = [] 
    labels = []
    model.eval()
    for batch in validation_dataloader :
    # Add batch to GPU
    # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)
        b_labels = b_labels.to(device)
    # Telling the model not to compute or store gradients
        with torch.no_grad():
      # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        # Move logits and labels to CPU
        logits = logits[0].to('cpu')
        label_ids = b_labels.to('cpu')
        #Risultato sul set di test
        #Strato finale di decisione sui risultati tramite una sigmoid function
        labelss = np.argmax(label_ids,axis=1)
        b_predictions = np.argmax(logits,axis=1)
        predictions.extend(b_predictions.tolist())
        labels.extend(labelss.tolist())
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Calcola l'accuratezza globale
    accuracy_global = accuracy_score(labels, predictions)

    # Calcola la precisione, il richiamo, e l'F1 globali
    precision_global = precision_score(labels, predictions, average='weighted')
    recall_global = recall_score(labels, predictions, average='weighted')
    f1_global = f1_score(labels, predictions, average='weighted')

    # Calcola la precisione, il richiamo, e l'F1 per ciascuna classe
    precision_per_class = precision_score(labels, predictions, average=None)
    recall_per_class = recall_score(labels, predictions, average=None)
    f1_per_class = f1_score(labels, predictions, average=None)

    # Stampa le metriche
    print("Accuratezza globale:", accuracy_global)
    print("Precisione globale:", precision_global)
    print("Richiamo globale:", recall_global)
    print("F1 globale:", f1_global)

    #    Stampa le metriche per classe
    for i in range(len(precision_per_class)):
        print(f"\nClasse {i+1}:")
        print("Precisione:", precision_per_class[i])
        print("Richiamo:", recall_per_class[i])
        print("F1:", f1_per_class[i])

    # Genera il report di classificazione
    print("\nReport di classificazione:")
    print(classification_report(labels, predictions))
'''

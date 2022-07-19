# !pip install kaggle pandas

from pickletools import optimize
from unicodedata import name
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, TFAutoModel
import numpy as np
import tensorflow as tf

tf.config.experimental.list_physical_devices("GPU")

# api = KaggleApi()
# api.authenticate()

# api.competition_download_file(
#     "sentiment-analysis-on-movie-reviews", "train.tsv.zip", path="./"
# )

# * Get CSV Data
df = pd.read_csv("./train.tsv", sep="\t")

# * Get Data Info
df.drop_duplicates(subset="SentenceId", keep="first", inplace=True)
print(df.head())
print(len(df))

# * Display a cool graph
seqlen = df["Phrase"].apply(lambda x: len(x.split()))

sns.set_style("darkgrid")
plt.figure(figsize=(16, 10))
sns.displot(seqlen)

# * Get the tokenizer
SEQ_LEN = 50

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

Xids = np.zeros((len(df), SEQ_LEN))
Xmask = np.zeros((len(df), SEQ_LEN))

print("Shape: ", Xids.shape)

# * Encode input data
for i, sequence in enumerate(df["Phrase"]):
    tokens = tokenizer.encode_plus(
        sequence,
        max_length=SEQ_LEN,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors="tf",
    )
    Xids[i, :], Xmask[i, :] = tokens["input_ids"], tokens["attention_mask"]

print("Xids: ", Xids)
print("Xmask: ", Xmask)

print("Sentiments: ", df["Sentiment"].unique())

arr = df["Sentiment"].values
labels = np.zeros((arr.size, arr.max() + 1))

labels[np.arange(arr.size), arr] = 1
print("Labels: ", labels)

# * Deallocate memory to SSD for later use
with open("xids.npy", "wb") as f:
    np.save(f, Xids)
with open("xmask.npy", "wb") as f:
    np.save(f, Xmask)
with open("labels.npy", "wb") as f:
    np.save(f, labels)

del df, Xids, Xmask, labels

with open("xids.npy", "rb") as fp:
    Xids = np.load(fp)
with open("xmask.npy", "rb") as fp:
    Xmask = np.load(fp)
with open("labels.npy", "rb") as fp:
    labels = np.load(fp)


dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))


for i in dataset.take(1):
    print(i)
    print(i[0].shape)
    print(i[1].shape)
    print(i[2].shape)
    break


def map_function(input_ids, masks, labels):
    return {"input_ids": input_ids, "attention_mask": masks, "labels": labels}


dataset = dataset.map(map_function)

for i in dataset.take(1):
    print(i)
    break

dataset = dataset.shuffle(100000).batch(32)

DS_LEN = len(list(dataset))
print("Dataset Length: ", DS_LEN)

SPLIT = 0.9

train = dataset.take(round(DS_LEN * SPLIT))
val = dataset.skip(round(DS_LEN * SPLIT))

del dataset

bert = TFAutoModel.from_pretrained("bert-base-uncased")

input_ids = tf.keras.Input(shape=(SEQ_LEN,), name="input_ids", dtype="int32")
mask = tf.keras.Input(shape=(SEQ_LEN,), name="attention_mask", dtype="int32")

embeddings = bert(input_ids, attention_mask=mask, return_tensors="tf")[0]

# * Here we can experiment with different layers

X = tf.keras.layers.GlobalAveragePooling1D()(embeddings)
X = tf.keras.layers.BatchNormalization()(X)
X = tf.keras.layers.Dense(128, activation="relu")(X)
X = tf.keras.layers.Dropout(0.1)(X)
X = tf.keras.layers.Dense(32, activation="relu")(X)
y = tf.keras.layers.Dense(5, activation="softmax", name="outputs")(X)

model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

print("Summary", model.summary())

# * Freeze layer
model.layers[2].trainable = False


optimizer = tf.keras.optimizers.Adam(0.01)
loss = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy("accuracy")


model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

history = model.fit(train, epochs=50, validation_data=val)

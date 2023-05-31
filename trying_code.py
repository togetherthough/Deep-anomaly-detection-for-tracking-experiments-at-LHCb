import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from models.VariationalAutoencoder import VAE
from sklearn.metrics import accuracy_score
print(tf.version.VERSION)

df = pd.read_csv('creditcard.csv')


target_feature = "Class"
# instantiate two more datasets one to contain the normal data
# and the other to hold the anomalous instances
normal_data = df[df[target_feature] == 0].drop(columns=target_feature)
anomalous_data = df[df[target_feature] == 1].drop(columns=target_feature)

# also instantiate a dataframe to carry all the features
# and a second one to instead contain the target variable
features = df.drop([target_feature], axis=1)
label = df[target_feature]

# the stratify parameter here is set to be "isghost" meaning that the
# proportions of ghost and normal tracks in both train and test datasets
# will be kept the same...
x_train, x_test, y_train, y_test = train_test_split(
    features, label, test_size=0.2, stratify=label
)

# remember that at least for the autoencoders we're gonna use just the
# normal instances during the training phase so we have to filter those in x_train
train_index = y_train[y_train == 0].index
train_normal_data = x_train.loc[train_index]

latent_dim = round(train_normal_data.shape[1] / 2)


optimizer = tf.optimizers.Adam(learning_rate=0.003)
anomaly_detector = VAE(
    input_dim=train_normal_data.shape[1], latent_dim=latent_dim)

anomaly_detector.compile(optimizer=optimizer)



# the model needs to be trained on the normal data to learn the underlying distribution
anomaly_detector.fit(train_normal_data, epochs=2, batch_size=64, shuffle=True)
print("Training was successful")

predictions = anomaly_detector.predict(x_test)
print("Anomaly scores computed")

# here we're relying on a threshold set just based on the percentile
threshold = np.percentile(predictions, 95)

predictions = np.where(predictions > threshold, 1, 0)

# print the accuracy and the roc
accuracy = accuracy_score(y_test, predictions)

print("Accuracy = {}".format(accuracy_score(y_test, predictions)))
print("Precision = {}".format(precision_score(y_test, predictions)))
print("Recall = {}".format(recall_score(y_test, predictions)))

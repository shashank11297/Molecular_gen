from model_archi import *
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import MolsToGridImage
import numpy as np
import tensorflow as tf
from tensorflow import keras



csv_path = tf.keras.utils.get_file(
    "qm9.csv", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
)

data = []
with open(csv_path, "r") as f:
    for line in f.readlines()[1:]:
        data.append(line.split(",")[1])


adjacency_tensor, feature_tensor = [], []
for smiles in data: # data[::10] - We have only used 10th of the dataset to save time, data - else use full
    adjacency, features = smiles_to_graph(smiles)
    adjacency_tensor.append(adjacency)
    feature_tensor.append(features)

adjacency_tensor = np.array(adjacency_tensor)
feature_tensor = np.array(feature_tensor)

wgan = GraphWGAN(generator, discriminator, discriminator_steps=1)

wgan.compile(
    optimizer_generator=keras.optimizers.Adam(5e-4),
    optimizer_discriminator=keras.optimizers.Adam(5e-4),
)

training_results = wgan.fit([adjacency_tensor, feature_tensor], epochs=25, batch_size=64)

wgan.save_weights('model_Weights.h5')

print(training_results.history['loss_gen'])
print(training_results.history['loss_dis'])
import argparse
from pathlib import Path

import xlrd as x
import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.merge import _Merge
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K
from functools import partial

BATCH_SIZE = 64
TRAINING_RATIO = 5
GRADIENT_PENALTY_WEIGHT = 10
DATA_FILE = "../data/augeralle.xlsx"


def reCol(file, c):
    '''
    Methode zur Erfassung und Rückgabe der Ereignisdaten aus einer Exceltabelle
    '''
    wb = x.open_workbook(file)
    s = wb.sheet_by_index(0)
    col = np.asarray([s.cell(c, r).value for r in range(1, s.ncols)])
    return col


def reNRows(file, nS):
    wb = x.open_workbook(file)
    s = wb.sheet_by_index(nS)
    return s.nrows


def wasserstein_loss(y_true, y_pred):
    """ Berechnet Wasserstein loss für ein Batch.
        Das Wasserstein loss wird simplerweise über eine
        lineare Aktivierungsfunktion berechnet.
        Kann negativ sein.
    """
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    """ Berechnet das gpl für ein normalisiertes Batch. """
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


def make_generator():
    """Erzeugt ein Generatormodell, dass einen 6-dimensionalen Noisetensor als In nimmt und
       ein ebenfalls 6-dimensionalen tensor als Out liefert."""
    model = Sequential()
    model.add(Dense(180, input_dim=6))
    model.add(LeakyReLU())
    model.add(Dense(45))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(29))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(20))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(15))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # Da die Inputdaten in das Intervall [-1, 1] gepresst wurden, liegt nah, 
    # die Tangens-Hyperbolicus Funktion für die Outputdaten des Generators zu nutzen, 
    # damit die jeweiligen Intervalle identisch sind
    model.add(Dense(6, activation='tanh'))
    return model


def make_discriminator():
    """Erzeugt ein Diskriminatormodell, dass ein 6-dimensionalen tensor als In nimmt und einen Wert ausgibt,
       der angibt, ob die Eingabe generiert oder echt ist."""
    model = Sequential()
    model.add(Dense(240, input_dim=6))
    model.add(LeakyReLU())
    model.add(Dense(120))
    model.add(LeakyReLU())
    model.add(Dense(60))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(30))
    model.add(LeakyReLU())
    model.add(Dense(1, kernel_initializer='he_normal'))
    return model


class RandomWeightedAverage(_Merge):
    """Zufällige Zuweisung von Synapsengewichten"""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


def generate_events(generator_model, output_dir, epoch):
    """Zufallsrauschen in Form eines 6-dimensionalen Vektors wird übergeben"""
    test_event_tensor = generator_model.predict(np.random.rand(10, 100))
    test_event_tensor = np.squeeze(np.round(test_event_tensor).astype(np.uint8))
    # tiled_event_output =

    # test_tensor = generator_model.predict(np.random.rand(10, 100))
    # test_tensor = np.squeeze(np.round(test_tensor).astype(np.uint8))
    # tiled_output = test_tensor(test_tensor)
    # outfile = os.path.join(output_dir, 'epoch_{}.txt'.format(epoch))
    # tiled_output.save(outfile)


parser = argparse.ArgumentParser(description="Improved Wasserstein GAN "
                                             "implementation for Keras.")
parser.add_argument("--output_dir", "-o", required=True,
                    help="Directory to output generated files to")
args = parser.parse_args()

# Daten in leere Liste laden

# OUTPUT_FILE = open('out.txt', 'w')
einHalbEvents = []
einHalbEvents2 = []
for i in range(1, int((reNRows(DATA_FILE, 0))/2)):
    ROW_I = reCol(DATA_FILE, i)
    # OUTPUT_FILE.write(ROW_I + "\n")
    print(ROW_I)

    einHalbEvents.append(ROW_I)
for i in range(int((reNRows(DATA_FILE, 0)/2)), reNRows(DATA_FILE, 0)):
    ROW_I = reCol(DATA_FILE, i)
    print(ROW_I)
    einHalbEvents2.append(ROW_I)
alleEvents = np.concatenate((einHalbEvents, einHalbEvents2), axis=0)
alleEvents = (alleEvents.astype(np.float32)-141) /141

# Initialisierung von Generator und Diskriminator
generator = make_generator()
discriminator = make_discriminator()

for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False
generator_input = Input(shape=(100,))
generator_layers = generator(generator_input)
discriminator_layers_for_generator = discriminator(generator_layers)
generator_model = Model(inputs=[generator_input],
                        outputs=[discriminator_layers_for_generator])
# Adam als Optimizer!
generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                        loss=wasserstein_loss)

# Nach der vollständigen Initialisierung der beiden NNs können sie jetzt trainiert werden
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False

# Diskriminator kriegt sowohl echte Ereignisse, als auch Noise vom Gen.
real_samples = Input(shape=alleEvents.shape[1:])
generator_input_for_discriminator = Input(shape=(100,))
generated_samples_for_discriminator = generator(generator_input_for_discriminator)
discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
discriminator_output_from_real_samples = discriminator(real_samples)

# Gewichtung muss angepasst werden --> samples werden zusammengefasst
averaged_samples = RandomWeightedAverage()([real_samples,
                                            generated_samples_for_discriminator])
# aS werden überschrieben --> durch das NN
averaged_samples_out = discriminator(averaged_samples)

# Grdients müssen für die Loss-Funktion noch definiert werden
partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)

partial_gp_loss.__name__ = 'gradient_penalty'

# Keras ist sogar so picky, dass es erfordert, dass die gleiche Anzahl von in und out samples
# durch das System laufen
# Drei Outs, ein echtes, ein generiertes und ein zusammengefasstes Sample
discriminator_model = Model(inputs=[real_samples,
                                    generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])

# Adam für alles, Wasserstein für echt und generiert und gp loss für die averaged samples
discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])

# label Vektoren werden erstellt, dummy_y wird zum gp loss übergeben und nicht benutzt
positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

# Trainingsfunktion
for epoch in range(100):
    np.random.shuffle(alleEvents)
    print("Epoch: ", epoch)
    print("Number of batches: ", int(alleEvents.shape[0] // BATCH_SIZE))
    discriminator_loss = []
    generator_loss = []
    minibatches_size = BATCH_SIZE * TRAINING_RATIO
    for i in range(int(alleEvents.shape[0] // (BATCH_SIZE * TRAINING_RATIO))):
        discriminator_minibatches = alleEvents[i * minibatches_size:
                                               (i + 1) * minibatches_size]
        for j in range(TRAINING_RATIO):
            image_batch = discriminator_minibatches[j * BATCH_SIZE:
                                                    (j + 1) * BATCH_SIZE]
            noise = np.random.rand(BATCH_SIZE, 100).astype(np.float32)
            discriminator_loss.append(discriminator_model.train_on_batch(
                [image_batch, noise],
                [positive_y, negative_y, dummy_y]))
        generator_loss.append(generator_model.train_on_batch(np.random.rand(BATCH_SIZE,
                                                                            100),
                                                             positive_y))
    generate_events(generator, args.output_dir, epoch)


# Zusammenführung der NNs
def re_wgan():
    model = Sequential()
    model.add(generator_model)
    model.add(discriminator_model)
    return model


# Framework kompilieren
re_wgan.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Framework als JSON darstellen
model_json = re_wgan().to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# Synapsengewichte übergeben
re_wgan.save_weights("model.h5")
print("Saved model to disk")

'''
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
'''

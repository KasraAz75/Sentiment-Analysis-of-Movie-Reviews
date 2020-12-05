#!/usr/bin/env python
# SentimentAnalysis.py

from tensorflow.keras import layers
from tensorflow.keras import models
import matplotlib.pyplot as plt
from DataLoader import load_data


def build_model(n_words=10000, dim_embedding=64):
    """ Model's architecture to implement "Sentiment Analysis". """

    model = models.Sequential()
    model.add(layers.Embedding(n_words, dim_embedding))
    model.add(layers.Dropout(0.3))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def main():
    """ __main__ """

    # Compiling the model
    model = build_model()
    model.summary()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Loading the datasets
    train_dataset, valid_dataset, test_dataset = load_data(path="imdb.pkl", n_words=100000,
                                                           valid_portion=0.1, maxlen=None, sort_by_len=True)

    # Training
    EPOCHS = 10
    BATCH_SIZE = 16
    history = model.fit(train_dataset,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    validation_data = valid_dataset
    )

    # Validating
    loss_acc, test_acc = model.evaluate(test_dataset, batch_size=BATCH_SIZE)

    # Results
    return history, print('Test accuracy:', test_acc)


def plot_results():
    """ Plotting the result of training """

    history_dict = main[0].history

    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

class MNISTNeuralNetwork():
    def __init__(self, hidden_units):
        self.input = tf.keras.Input(shape=(28,28), name="input")
        self.flat = tf.keras.layers.Flatten(name="flatten")(self.input)
        self.hidden = tf.keras.layers.Dense(units=hidden_units,activation='relu', name="hidden_1")(self.flat)
        self.output = tf.keras.layers.Dense(units=10, activation="softmax", name="output")(self.hidden)

    def build(self):
        print("\nBuilding model...")
        self.model = tf.keras.Model(inputs=self.input, outputs=self.output)
        return self.model
    
    def train(self, x, y, epochs):
        optimizer = tf.keras.optimizers.SGD()
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        call_back = tf.keras.callbacks.ModelCheckpoint(filepath="best_model.h5",
                                                        save_weights_only=True,
                                                        monitor='loss',
                                                        mode='min',
                                                        save_best_only=True)
        self.model.compile(optimizer=optimizer, loss=loss)
        hist = self.model.fit(x, y, epochs=epochs, callbacks=[call_back])
        return hist
    
    def save(self, path):
        print(f"Save model to {path}...")
        self.model.save_weights(path)

    def load(self, checkpoint):
        print(f"Load model from {checkpoint}...")
        self.model.load_weights(checkpoint)

    def predict(self, x):
        pred = self.model.predict(x, verbose=0)
        label = tf.math.argmax(pred, axis=1).numpy()
        return label

    def evaluate(self, X_test, y_test):
        print("Evaluate model...")
        pred = self.model.predict(X_test, verbose=1)
        pred_label = tf.math.argmax(pred, axis=1).numpy()
        print(f"\tAccuracy: {accuracy_score(y_test, pred_label):.3f}")
        print(f"\tPrecision: {precision_score(y_test, pred_label, average='micro'):.3f}")
        print(f"\tRecall: {recall_score(y_test, pred_label, average='micro'):.3f}")
        print(f"\tF1-score: {f1_score(y_test, pred_label, average='micro'):.3f}")

        confusion_m = confusion_matrix(y_test, pred_label)
        
        cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_m)
        print("\nDisplay Confusion Matrix:\n")
        cm_display.plot()
        plt.show()

    def summary(self):
        print(self.model.summary())
    
    def get_trained_params(self):
        weights = self.model.get_weights()
        return weights

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    print("\n\t**Show the data**")
    idx = 1
    plt.figure(figsize=(10, 10))
    plt.suptitle("Show the data")
    for i in range(5):
        for j in range(5):
            x_idx = tf.experimental.numpy.random.randint(low=0,
                                            high=X_train.shape[0],
                                            size=1,dtype=tf.experimental.numpy.int64).numpy()[0]
            plt.axis('off')
            plt.subplot(5,5, idx)
            plt.imshow(X_train[x_idx],cmap="gray")
            plt.title(f"Label: {y_train[x_idx]}")
            idx += 1
    plt.show()

    X_train, X_test = X_train/255., X_test/255.

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    print("\n\t**Build Model**")
    model = MNISTNeuralNetwork(28*28)
    model.build()
    model.summary()
    print()

    print("\n\t**Train Model on Train Set**")
    hist = model.train(X_train, y_train, epochs=10)
    plt.plot(hist.history["loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

    print("\n\t**Test Model on Test Set**")
    model.evaluate(X_test, y_test)

    print("\n\t**Save Model**")
    model.save("checkpoint.h5")

    print("\n\t**Load Model**")
    model.load("best_model.h5")

    print("\n\t**Get Model's parameters**")
    weights = model.get_trained_params()
    print()

    print("\n\t**Demo on Image**")
    idx = 1
    plt.figure(figsize=(10, 10))
    plt.suptitle("Prediction on Test Set Visualization")
    for i in range(5):
        for j in range(5):
            x_idx = tf.experimental.numpy.random.randint(low=0,
                                            high=X_test.shape[0],
                                            size=1,dtype=tf.experimental.numpy.int64).numpy()[0]
            plt.axis('off')
            plt.subplot(5,5, idx)
            plt.imshow(X_test[x_idx],cmap="gray")
            plt.title(f"Pred: {model.predict(tf.expand_dims(X_test[x_idx], axis=0))[0]}")
            idx += 1
    plt.show()
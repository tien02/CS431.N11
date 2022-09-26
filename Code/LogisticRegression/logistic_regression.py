import numpy as np
import tensorflow as tf
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification

class Logistic_Regression():
    def __init__(self, input_channels, output_channels):
        self.input = tf.keras.Input(shape=(input_channels,))
        self.output = tf.keras.layers.Dense(units=output_channels, activation="sigmoid")(self.input)

    def build(self):
        print("Building model...")
        self.model = tf.keras.Model(inputs=self.input, outputs=self.output)
        return self.model
    
    def train(self, x, y, epochs):
        optimizer = tf.keras.optimizers.SGD()
        loss = tf.keras.losses.BinaryCrossentropy()
        call_back = tf.keras.callbacks.ModelCheckpoint(filepath="logistic_regression_pro.h5",
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

    def predict(self, x, threshold=0.5):
        pred = self.model.predict(x)
        label = pred > threshold
        label = label.astype('int')
        return label

    def evaluate(self, X_test, y_test, threshold=0.5):
        print("Evaluate model...")
        pred = self.model.predict(X_test)
        label = pred > threshold
        label = label.astype('int')
        print(classification_report(y_test, label))

    def summary(self):
        print(self.model.summary())
    
    def get_trained_params(self):
        weights = self.model.get_weights()
        w = weights[0]
        b = weights[1]
        param = {"w": w, "b": b}
        pprint(param)
        return param

if __name__ == '__main__':
    print("\t**Make Data**")
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1, random_state=42)
    plt.title("Data")
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=25, edgecolor="k")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    plt.show()

    print("\n\t**Build Model**")
    model = Logistic_Regression(2, 1)
    model.build()
    model.summary()

    print("\n\t**Split Train-Test Set**")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    print("\n\t**Train Model on Train Set**")
    hist = model.train(X_train, y_train, epochs=50)
    plt.plot(hist.history["loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

    print("\n\t**Test Model on Test Set**")
    model.evaluate(X_test, y_test)

    print("\n\t**Build Model**")
    model.save("checkpoint.h5")

    print("\n\t**Save Model**")
    model.load("./logistic_regression_pro.h5")

    print("\n\t**Get Model's parameters**")
    weights = model.get_trained_params()
    w = weights["w"][0][0]
    b = weights["b"][0]

    print("\n\t**Plot model on Test Set**")
    xm = np.arange(-4, 4, 0.05)
    xlen = len(xm)
    ym = np.arange(-4, 4, 0.05)
    ylen = len(ym)
    xx, yy = np.meshgrid(xm, ym)

    xx1 = xx.ravel().reshape(1, xx.size)
    yy1 = yy.ravel().reshape(1, yy.size)

    XX = np.concatenate((xx1, yy1), axis = 0).T

    Z = model.predict(XX)

    Z = Z.reshape(xx.shape)

    CS = plt.contourf(xx, yy, Z, 200, cmap='jet', alpha = .1)

    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xticks(())
    plt.yticks(())
    plt.scatter(X_test[:, 0], X_test[:, 1], marker="o", c=y_test, s=25, edgecolor="k")
    plt.show()

    print("\n=== Finish ===")
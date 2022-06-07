import numpy as np
from ventana_ui import *
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.cant_gen
        self.ejec_algebra.clicked.connect(self.perceptron)
        self.ejec_tensor.clicked.connect(self.tensor)
        self.abalone_features = []
        self.abalone_labels = []
        self.leer_Archivo()

    
    def perceptron(self):
        pass

    def tensor(self):
        n = 0.1
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error',
                      optimizer=tf.keras.optimizers.Adam(n))
        historial = model.fit(self.abalone_features,
                              self.abalone_labels, epochs=int(self.cant_gen.text()))
        result = model.predict(self.abalone_features)
        print(f'PESOS: {model.get_weights()}')
        plt.plot(historial.history['loss'], label=f'n={n}')
        print(f'RESULT: {result}')
        plt.legend()
        plt.xlabel("Generaciones")
        plt.ylabel("Rango error")
        plt.show()

    def leer_Archivo(self):
        abalone_train = pd.read_csv(
            "data.csv", names=["x1", "x2", "x3", "x4", "y"])
        self.abalone_features = abalone_train.copy()
        self.abalone_labels = self.abalone_features.pop("y")
        self.abalone_features = np.array(self.abalone_features)
        print("lista de x: ", self.abalone_features)
        print("lista de y: ", self.abalone_labels)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()

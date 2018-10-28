from keras.layers import Input, Dense, Activation, Dropout
from keras.models import Model


def model_keras(input_shape):
    X_input = Input(shape=(input_shape,))
    X = Dense(124, name="dense1")(X_input)
    X = Activation("relu")(X)
    X = Dense(256, name="dense2")(X)
    X = Activation("relu")(X)
    X = Dropout(0.5, name="dropout1")(X)
    X = Dense(128, name="dense3")(X)
    X = Activation("relu")(X)
    X = Dropout(0.5, name="dropout2")(X)
    X = Dense(64, name="dense4")(X)
    X = Activation("relu")(X)
    X = Dropout(0.5, name="dropout3")(X)
    X = Dense(14, name="dense5")(X)
    X = Activation("softmax")(X)
    model_kerass = Model(inputs=X_input, outputs=X, name="kubi_model")
    # model = keras.Sequential()
    # model.add(keras.layers.Dense(128, input_dim=14, activation="relu"))
    # model.add(keras.layers.Dense(256, activation="relu"))
    # model.add(keras.layers.Dropout(0.5))
    # model.add(keras.layers.Dense(128, activation="relu"))
    # model.add(keras.layers.Dropout(0.5))
    # model.add(keras.layers.Dense(64, activation="relu"))
    # model.add(keras.layers.Dropout(0.5))
    # model.add(keras.layers.Dense(14))
    # model.add(keras.layers.Activation("softmax"))
    return model_kerass

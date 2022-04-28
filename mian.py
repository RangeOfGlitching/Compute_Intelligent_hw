import numpy as np
import load_testData as testD
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt

test_in_path = r"TestData\IN.dat"
test_OUT_path = r"TestData\OUT.dat"

Data = testD.MlpData(test_in_path, test_OUT_path)

InData = Data.InData()
outData = Data.outData()

T = 4
InData = np.array(InData).reshape(-1, T, 1)  # Now the data should be N x T x D
outData = np.array(outData)
N = len(InData)

### try autoregressive RNN model
i = Input(shape=(T, 1))
x = SimpleRNN(15, activation='relu')(i)
x = Dense(1)(x)
model = Model(i, x)
model.compile(
    loss='mse',
    optimizer=Adam(lr=0.001),
)
r = model.fit(
    InData[:-N // 2], InData[:-N // 2],
    epochs=80,
    validation_data=(InData[-N // 2:], outData[-N // 2:]),
)

# Plot loss per iteration

plt.plot(r.history['loss'], label='loss', color="black")
plt.plot(r.history['val_loss'], label='val_loss', color="blue")

plt.legend(["loss", "val_loss"], loc="lower right")
plt.show()

# Forecast future values (use only self-predictions for making future predictions)

validation_target = outData[-N // 2:]
validation_predictions = []

# first validation input
last_x = InData[-N // 2]  # 1-D array of length T

while len(validation_predictions) < len(validation_target):
    p = model.predict(last_x.reshape(4, 1))[0, 0]  # 1x1 array -> scalar

    # update the predictions list
    validation_predictions.append(p)

    # make the new input
    last_x = np.roll(last_x, -1)
    last_x[-1] = p
plt.plot(validation_target, label='forecast target')
plt.plot(validation_predictions, label='forecast prediction')
plt.legend()
plt.show()

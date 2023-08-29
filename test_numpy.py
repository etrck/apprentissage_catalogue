model = keras.models.Sequential()
model.add( keras.layers.InputLayer(input_shape=(41, 47)))
model.add( keras.layers.LSTM(50, return_sequences=False, activation='relu'))
model.add( keras.layers.Dense(1))
model.summary()

model.compile(optimizer='rmsprop', loss='mse', metrics = ['mae'])

history=model.fit(train_generator, epochs = 5, validation_data = test_generator)
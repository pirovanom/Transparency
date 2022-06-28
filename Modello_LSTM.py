# multivariate data preparation
from pickle import TRUE
import numpy as np
import pandas as pd
from numpy import array
from pandas import read_csv
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras import optimizers
#import tensorflow as tf

#preparo il numpy array time
time = list()
time = read_csv('/home/marta/python/7-LSTM_prova2/time.csv', header=0, index_col=0)
righe_time = len(time.index)
print("righe time: ", righe_time)

# definisco i dati di input in un dataframe, che converto in un numpy array con 8 colonne tante sono le features
df = read_csv('/home/marta/python/7-LSTM_prova2/input_data.csv', header=0, index_col=0)
#print(df)
#print(type(df), "")
#divido i dati di train e di test
righe_tot = len(df.index)
percentuale = 0.8           #percentuale di dati_x riservata al train, gli altri sono riservati al train
separatore = int(percentuale * righe_tot)
righe_train = separatore
righe_test = righe_tot - separatore
train = df.iloc[:separatore,:]
test = df.iloc[separatore:,:]
#time_train e time_test
time_train = time.iloc[:separatore,:]
time_test = time.iloc[separatore:,:]
time_train.drop(time_train.index[[0, 1]], inplace=True) 
time_test.drop(time_test.index[[0, 1]], inplace=True) 

print("righe_tot:  ", righe_tot)
print("righe train: ", len(train.index))
print("righe test:  ", len(test.index))
print("")
#converto i dati di input in un array numpy
train=train.to_numpy()
test=test.to_numpy()
time_train=time_train.to_numpy()
time_test=time_test.to_numpy()
print("train.shape:  ", train.shape, "   test.shape:  ", test.shape)
print("time_train.shape:  ", time_train.shape, "   time_test.shape:  ", time_test.shape)
print("")

# definisco una funzione che splitta i dati
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# splitto i dati
n_steps = 3
x_train, y_train = split_sequences(train, n_steps)
x_test, y_test = split_sequences(test, n_steps)
print("X_train.shape:  ", x_train.shape, "   y_train.shape:  ", y_train.shape)
print("X_test.shape:  ", x_test.shape, "   y_test.shape:  ", y_test.shape)
print("")
# summarize the data
#for i in range(len(X_train)):
"""for i in range(5):
	print(X_train[i], y_train[i])
print("")
for i in range(5):
	print(X_test[i], y_test[i])"""

# definisco il modello
n_features = 7
model = Sequential()
model.add(Bidirectional(LSTM(128, activation='tanh', return_sequences=True), input_shape=(n_steps, n_features)))
model.add(Bidirectional(LSTM(16, activation='tanh', return_sequences=True)))
model.add(Bidirectional(LSTM(8, activation='tanh')))
model.add(Dense(1, activation='sigmoid'))

#compilo il modello
from tensorflow import keras
opt = keras.optimizers.Adam(learning_rate = 0.0001)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=["accuracy"])
"""model.compile(
    loss='mean_squared_error',      #specifico la loss function
    optimizer='adam',               #specifico l'ottimizzatore (stochastic gradient descent)
    #metrics=["accuracy"],          #specifico la metrica
)"""

#alleno il modello
"""model.fit(
    x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=1
)
model.fit(
    x_train, y_train, validation_data=(x_validate, y_validate), batch_size=64, epochs=10
)"""

model.fit(x_train, y_train, epochs=3)
#testo il modello
predictions = model.predict(x_test, verbose=0)
print("Risultati: ")
#for i in range(righe_test - n_steps + 1):
print("i,   predictions,   test")
for i in range(10):
	print(i+1, predictions[i], y_test[i])
"""for i in range(righe_test - n_steps + 1):
    result = tf.argmax(model.predict(tf.expand_dims(x_test[i],0)), axis=1)      #attiviamo il modello
    print(i+1, result.numpy(), y_test[i]) """         

#faccio il grafico
plt.figure(figsize=(12,6))
plt.scatter(time_test, predictions, label="Predictions")		#aggiungere sull'asse delle x il time giusto
plt.scatter(time_test, y_test, label="Test")
plt.scatter(time_train, y_train, label="Train")
plt.xlabel("Time [days]")
plt.ylabel("Normalized mean transparency")
plt.title(f"Predictions vs test")
legend = plt.legend(['Predictions','Test','Train'], title = "Legend")
plt.savefig("/home/marta/python/7-LSTM_prova2/Predictions_vs_Test.png")
plt.show()

#plot per fill
#dataframe per i plot
dataframe = pd.DataFrame(time_test)
dataframe=dataframe.assign(predict = 0)
dataframe.predict= pd.DataFrame(predictions)
dataframe=dataframe.assign(test = 0)
dataframe.test= pd.DataFrame(y_test)
dataframe.columns = ['time', 'predictions', 'test']
dataframe.set_index('time', drop = False, inplace = True)
#print(dataframe)
#print(df)
#print("")
#print(test)
dataframe.to_csv('/home/marta/python/7-LSTM_prova2/output.csv')
dataframe = dataframe.merge(df, how='inner', left_index=True, right_index=True)
#print(dataframe)


#plot
fill_nums = dataframe.fill_num.unique()
#print(fill_nums)
for k in (fill_nums):
	new_df = dataframe[dataframe.fill_num == k]
	#new_df.plot(kind='scatter', x='time', y='predictions')
	#new_df.plot(kind='scatter', x='time', y='test')
	new_df[['predictions','test']].plot()
	#plt.scatter(new_df['time'], new_df['predictions'])
	#plt.scatter(new_df['time'], new_df['test'])
	plt.xlabel("Time [days]")
	plt.ylabel("Normalized mean transparency")
	plt.title(f"Predictions vs test   -   Fill {k}")
	legend = plt.legend(['Predictions','Test'], title = "Legend")
	plt.savefig(f"/home/marta/python/7-LSTM_prova2/Per_fill/Fill_{k}.png")
	#plt.show()

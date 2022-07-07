# DESCRIZIONE: questo programma utilizza una rete neurale artificiale ricorrente chiamata Long Short Term Memory (LSTM) 
# per prevedere l'andamento della trasparenza dei cristalli ECAL con vari input di partenza

#importo le librerie
import numpy as np
#from numpy import concatenate
import pandas as pd
from numpy import array
from pandas import read_csv
#from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import keras
from keras import activations
from keras import layers
from keras import backend as K

#preparo il numpy array time
time = list()
time = read_csv('/home/marta/python/7-LSTM_prova2/time.csv', header=0, index_col=0)
righe_time = len(time.index)
print("righe time: ", righe_time)

# definisco i dati di input in un dataframe, che convertirò in un numpy array con 8 colonne, tante sono le features
df = read_csv('/home/marta/python/7-LSTM_prova2/input_data.csv', header=0, index_col=0)
#scalo i dati di trasparenza
massimo_trasp = df['trasparenza'].max()
minimo_trasp = df['trasparenza'].min()
df.trasparenza = (df.trasparenza - minimo_trasp) / (massimo_trasp - minimo_trasp)
#scalo i dati di fill
massimo_fill = df['fill_num'].max()
minimo_fill = df['fill_num'].min()
df.fill_num = (df.fill_num - minimo_fill) / (massimo_fill - minimo_fill)
#plt.plot(df)
#plt.show()

#divido i dati di train e di test in nuovi dataframe
righe_tot = len(df.index)
percentuale = 0.8           #percentuale di dati_x riservata al train, gli altri sono riservati al train
separatore = int(percentuale * righe_tot)
righe_train = separatore
righe_test = righe_tot - separatore
train_df = df.iloc[:separatore,:]
test_df = df.iloc[separatore:,:]
#time_train e time_test
time_train = time.iloc[:separatore,:]
time_test = time.iloc[separatore:,:]
time_train.drop(time_train.index[[0, 1]], inplace=True) 		#droppo
time_test.drop(time_test.index[[0, 1]], inplace=True) 

#controllo che la divione si andata a buon fine
print("righe_tot:  ", righe_tot)
print("righe train: ", len(train_df.index))
print("righe test:  ", len(test_df.index))
print("")

#converto i dataframe con i dati di input in array numpy
train=train_df.to_numpy()
test=test_df.to_numpy()
time_train=time_train.to_numpy()
time_test=time_test.to_numpy()
#print("train.shape:  ", train.shape, "   test.shape:  ", test.shape)
#print("time_train.shape:  ", time_train.shape, "   time_test.shape:  ", time_test.shape)

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
#confronto = y_test
print("X_train.shape:  ", x_train.shape, "   y_train.shape:  ", y_train.shape)
print("X_test.shape:  ", x_test.shape, "   y_test.shape:  ", y_test.shape)
print("")

# definisco il modello
n_features = 7
model = Sequential()
#def clipped_relu(x):
#clip_relu = activations.relu(x_train, alpha=0.0, max_value = 1, threshold=0.0)
#model.add(layers.Activation(clip_relu))
#clipped_relu = activations.relu(x_train, max_value=1.0)
#activations.relu(x_train, alpha=0.0, max_value = 1, threshold=0.0)"""
def clip_relu(x, max_value=1.):
	return activations.relu(x, max_value = 1.0)
model.add(Bidirectional(LSTM(128, activation='tanh', return_sequences=True), input_shape=(n_steps, n_features)))
model.add(Bidirectional(LSTM(64, activation='tanh', return_sequences=True)))
model.add(Bidirectional(LSTM(32, activation='tanh')))
model.add(Dense(1, activation=clip_relu))

#compilo il modello
from tensorflow import keras
opt = keras.optimizers.Adam(learning_rate = 0.0001)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=["accuracy"])

#definisco earlystopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

#alleno il modello
"""model.fit(
    x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=1
)
model.fit(
    x_train, y_train, validation_data=(x_validate, y_validate), batch_size=64, epochs=10
)"""

history = model.fit(x_train, y_train, validation_split=0.15, batch_size=64, epochs=60, callbacks=[es])	#divide in automatico i dati di train in train e validation

#testo il modello
predictions = model.predict(x_test, verbose=0)

#riscalo i dati
for i in range (len(predictions)):
	predictions[i] = minimo_trasp + (predictions[i] * (massimo_trasp - minimo_trasp))
	y_test[i] = minimo_trasp + (y_test[i] * (massimo_trasp - minimo_trasp))

# stampo i risultati e li salvo in un file di output
print("Risultati: ")
print("i,   predictions,   test")
#for i in range(righe_test - n_steps + 1):
for i in range(10):
	print(i+1, predictions[i], y_test[i])        

# riempio un nuovo dataframe per i plot
dataframe = pd.DataFrame(time_test)
dataframe=dataframe.assign(predict = 0)
dataframe.predict= pd.DataFrame(predictions)
dataframe=dataframe.assign(test = 0)
dataframe.test= pd.DataFrame(y_test)
dataframe.columns = ['time', 'predictions', 'test']
dataframe.set_index('time', drop = False, inplace = True)
dataframe.to_csv('/home/marta/python/7-LSTM_prova2/output.csv')
dataframe = dataframe.merge(df, how='inner', left_index=True, right_index=True)
#print(type(df))
#print(df)
#print("Dataframe finale")
#print(dataframe)

#faccio il grafico con tutti i dati: train, test e predictions
"""plt.figure(figsize=(12,6))
plt.scatter(time_test, y_test, label="Test")
plt.scatter(time_test, predictions, label="Predictions")		
plt.scatter(time_train, y_train, label="Train")
plt.xlabel("Time [days]")
plt.ylabel("Normalized mean transparency")
plt.title(f"Predictions, test and train")
legend = plt.legend(['Test', 'Predictions', 'Train'], title = "Legend")
plt.savefig("/home/marta/python/7-LSTM_prova2/Predictions_Test_Train.png")
plt.show()"""

# faccio il grafico test vs predictions
plt.figure(figsize=(12,6))
plt.scatter(time_test, y_test, label="Test")
plt.scatter(time_test, predictions, label="Predictions")	
#plt.scatter(time_test, confronto, label="Confronto")
plt.xlabel("Time [days]")
plt.ylabel("Normalized mean transparency")
plt.title(f"Predictions vs test")
legend = plt.legend(['Test', 'Predictions', 'confronto'], title = "Legend")
plt.savefig("/home/marta/python/7-LSTM_prova2/Predictions_vs_Test.png")
plt.show()

# faccio il plot dei loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(f"Loss e val_loss")
plt.yscale("log")
legend = plt.legend(['loss','val_loss'], title = "Legend")
plt.savefig(f"/home/marta/python/7-LSTM_prova2/Loss_logScale.png")
plt.show()

# faccio il plot per singolo fill
dataframe.fill_num = minimo_fill + (dataframe.fill_num * (massimo_fill - minimo_fill))
fill_nums = dataframe.fill_num.unique()
for k in (fill_nums):
	new_df = dataframe[dataframe.fill_num == k]
	#new_df.plot(kind='scatter', x='time', y='predictions')
	#new_df.plot(kind='scatter', x='time', y='test')
	#new_df[['predictions','test']].plot.scatter(markersize=10, linewidth=0.5)
	#plt.plot(new_df['time'], new_df['predictions'], markersize=3, linewidth=0.75)
	plt.plot(new_df.time, new_df.predictions, ".b-", markersize=3, linewidth=0.75, label="Predictions")
	plt.plot(new_df.time, new_df.test, ".g-", markersize=3, linewidth=0.75, label="Test")
	#plt.plot(new_df['time'], new_df['test'], markersize=3, linewidth=0.75)
	plt.xlabel("Time [days]")
	plt.ylabel("Normalized mean transparency")
	plt.title(f"Predictions vs test   -   Fill {k}")
	legend = plt.legend(['Predictions','Test'], title = "Legend")
	plt.savefig(f"/home/marta/python/7-LSTM_prova2/Per_fill/Fill_{k}.png")
	plt.show()

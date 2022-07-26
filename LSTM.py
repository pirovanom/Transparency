# DESCRIZIONE: questo programma utilizza una rete neurale artificiale ricorrente chiamata Long Short Term Memory (LSTM) 
# per prevedere l'andamento della trasparenza dei cristalli di ECAL

# Importo le librerie
import numpy as np
import pandas as pd
from numpy import array
from pandas import read_csv
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
import datetime
import matplotlib.dates as mdates

# Preparo il numpy array time (servirà alla fine)
time = list()
time = read_csv('/home/marta/python/7-LSTM_prova2/time.csv', header=0, index_col=0)
righe_time = len(time.index)
#print("righe time: ", righe_time)

# Preparo il numpy array fill_numbers (servirà alla fine)
fill_numbers = list()
fill_numbers = read_csv('/home/marta/python/7-LSTM_prova2/fill_n.csv', header=0, index_col=0)
righe_fill_numbers = len(fill_numbers.index)
#print("righe righe_fill_numbers: ", righe_fill_numbers)

# Importo i dati di input in un dataframe
df = read_csv('/home/marta/python/7-LSTM_prova2/input_data.csv', header=0, index_col=0, usecols = [0, 1, 2, 3, 4, 7, 8])
#print(df)

# Definisco le impostazioni per convertire il tempo da timestamp a yy/mm/dd hh:mm:ss

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
formatter.formats = ['%y',  # ticks are mostly years
					'%b',       # ticks are mostly months
					'%d',       # ticks are mostly days
					'%H:%M',    # hrs
					'%H:%M',    # min
					'%S.%f', ]  # secs
formatter.zero_formats = [''] + formatter.formats[:-1]
    # ...except for ticks that are mostly hours, then it is nice to have
    # month-day:
formatter.zero_formats[3] = '%d-%b'

formatter.offset_formats = ['',
	                        '%Y',
                            '%b %Y',
                            '%d %b %Y',
                            '%d %b %Y',
                            '%d %b %Y %H:%M', ]

# Scalo i dati di trasparenza
massimo_trasp = df['trasparenza'].max()
minimo_trasp = df['trasparenza'].min()
minimo_trasp = minimo_trasp - 0.001
df.trasparenza = (df.trasparenza - minimo_trasp) / (1 - minimo_trasp)
#plt.plot(df.trasparenza)
#plt.show()

# Divido i dati di train e di test in nuovi dataframe
righe_tot = len(df.index)
percentuale = 0.8          				    # percentuale di dati_x riservata al train, gli altri sono riservati al test
separatore = int(percentuale * righe_tot)	# riga a cui separo i dati
righe_train = separatore
righe_test = righe_tot - separatore
train_df = df.iloc[:separatore,:]			# creo il dataframe train_df
test_df = df.iloc[separatore:,:]			# creo il dataframe test_df

# Converto i dataframe con i dati di input in array numpy
train=train_df.to_numpy()
test=test_df.to_numpy()
#print("train.shape:  ", train.shape, "   test.shape:  ", test.shape)
#print("time_train.shape:  ", time_train.shape, "   time_test.shape:  ", time_test.shape)

# Definisco una funzione che trasforma i dati nella forma richiesta come input da LSTM 
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

# Trasformo i dati
n_steps = 3											# numero di timestep presi in input a ogni passo
x_train, y_train = split_sequences(train, n_steps)	# trasformo i dati di train
x_test, y_test = split_sequences(test, n_steps)		# trasformo i dati di test
#print("X_train.shape:  ", x_train.shape, "   y_train.shape:  ", y_train.shape)
#print("X_test.shape:  ", x_test.shape, "   y_test.shape:  ", y_test.shape)
#print("")

# Definisco il modello LSTM

n_features = 5						# Numero di variabili di input
model = Sequential()
def clip_relu(x):					# Definisco la funzione di attivazione per i layer di output
	return activations.relu(x, max_value = 1.01)
# Aggiungo due layer bidirezionali
model.add(Bidirectional(LSTM(64, activation='tanh', return_sequences=True), input_shape=(n_steps, n_features)))
model.add(Bidirectional(LSTM(32, activation='tanh')))
# Aggiungo il layer di output
model.add(Dense(1, activation=clip_relu))

# Compilo il modello

from tensorflow import keras
opt = keras.optimizers.Adam(learning_rate = 0.0001)				#definisco l'ottimizzatore (adam) e il learning rate
model.compile(optimizer=opt, loss='mean_squared_error')

# Definisco earlystopping (che interrompe l'allenamento quando i dati non migliorano per un certo numero (100) epoche)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)

# Alleno il modello
history = model.fit(x_train, y_train, validation_split=0.15, batch_size=64, epochs=1000, callbacks=[es])	
# divide in automatico i dati di train in train(85%) e validation(15%), 
# massimo epoche 1000, interrompe quando entra es (earlystopping)

# Testo il modello
#pred_train = model.predict(x_train, verbose=0)
predictions = model.predict(x_test, verbose=0)

# Riscalo i dati
diff_test = list()
#diff_train = list()
predictions = predictions.reshape(1891)
#pred_train = pred_train.reshape(7568)
for i in range (len(predictions)):
	predictions[i] = minimo_trasp + (predictions[i] * (1 - minimo_trasp))
	y_test[i] = minimo_trasp + (y_test[i] * (1 - minimo_trasp))
	#diff_test[i] = predictions[i] - y_test[i]

# Stampo i risultati e li salvo in un file di output

print("Risultati: ")
print("i,   predictions,   test")
#for i in range(righe_test - n_steps + 1):
for i in range(10):
	print(i+1, predictions[i], y_test[i])        

# Riempio un nuovo dataframe con tempo, predizioni, test e variabili di input per i plot

time_train = time.iloc[:separatore,:]
time_test = time.iloc[separatore:,:]
time_train.drop(time_train.index[[0, 1]], inplace=True) 		#droppo
time_test.drop(time_test.index[[0, 1]], inplace=True) 
train=train_df.to_numpy()
test=test_df.to_numpy()
time_train=time_train.to_numpy()
time_test=time_test.to_numpy()

dataframe = pd.DataFrame(time_test)
dataframe=dataframe.assign(predict = 0)
dataframe.predict= pd.DataFrame(predictions)
dataframe=dataframe.assign(test = 0)
dataframe.test= pd.DataFrame(y_test)
dataframe.columns = ['time', 'predictions', 'test']
#print(dataframe)
dataframe.set_index('time', drop = False, inplace = True)			# definisco il tempo come indice del dataframe
# Unisco dataframe e df 
dataframe = dataframe.merge(df, how='inner', left_index=True, right_index=True)
# Unisco dataframe e fill_numbers
dataframe = dataframe.merge(fill_numbers, how='inner', left_index=True, right_index=True)
print(dataframe)
# Converto il tempo da timestamp a yy/mm/dd hh:mm:ss
dataframe.time = [datetime.datetime.fromtimestamp(ts) for ts in dataframe.time]
dataframe.head()
print("Dataframe finale")
print(dataframe)
dataframe.to_csv('/home/marta/python/7-LSTM_prova2/output.csv')

# Faccio il grafico test vs predictions

plt.figure(figsize=(12,6))
plt.scatter(time_test, y_test, label="Test")
plt.scatter(time_test, predictions, label="Predictions")
#plt.scatter(time_train, y_train, label="Train")	
#plt.scatter(time_train, pred_train, label="Train predictions")
plt.xlabel("Time")
plt.xticks(rotation='45')			# ruota di 45 le scritte sull'asse delle x
plt.ylabel("Normalized mean transparency")
plt.title(f"Predictions vs test")
legend = plt.legend(['Test', 'Predictions', 'Train', 'Train predictions'], title = "Legend")
plt.savefig("/home/marta/python/7-LSTM_prova2/Predictions_vs_Test.png")
plt.show()

"""# Faccio il grafico delle differenze
plt.figure(figsize=(12,6))
plt.scatter(time_test, predictions - y_test, label="Test")
#plt.scatter(time_train, pred_train - y_train, label="Train")	
#plt.scatter(time_test, confronto, label="Confronto")
plt.xlabel("Time [days]")
plt.xticks(rotation='45')
plt.ylabel("Normalized mean transparency")
plt.title(f"Differenza")
legend = plt.legend(['Test', 'Train'], title = "Legend")
#plt.savefig("/home/marta/python/7-LSTM_prova2/Predictions_vs_Test.png")
plt.show()"""

# Faccio il plot dei loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(f"Loss e val_loss")
plt.yscale("log")
legend = plt.legend(['loss','val_loss'], title = "Legend")
plt.savefig(f"/home/marta/python/7-LSTM_prova2/Loss_logScale.png")
plt.show()

# Faccio il plot per singolo fill della trasparenza: predizioni e test 

# Definisco una lista contenente tutti i numeri di fill
fill_nums = dataframe.fill_num.unique()
# Per ogni fill faccio il grafico
for k in (fill_nums):
	new_df = dataframe[dataframe.fill_num == k]
	# Uso le impostazioni sul nuvo formato di data
	plt.gca().xaxis.set_major_formatter(formatter)
	plt.gca().xaxis.set_major_locator(locator)
	plt.plot(new_df.time, new_df.predictions, ".b-", markersize=3, linewidth=0.75, label="Predictions")
	plt.plot(new_df.time, new_df.test, ".g-", markersize=3, linewidth=0.75, label="Test")
	plt.xlabel("Time")
	#plt.xticks(rotation='45')
	plt.ylabel("Normalized mean transparency")
	plt.title(f"Predictions vs test   -   Fill {k}")
	legend = plt.legend(['Predictions','Test'], title = "Legend")
	plt.savefig(f"/home/marta/python/7-LSTM_prova2/Per_fill/Fill_{k}.pdf")
	plt.gcf().autofmt_xdate()
	plt.show()

# Descrizione: questo programma utilizza una rete neurale artificiale ricorrente chiamata Long Short Term Memory (LSTM) 
# per prevedere l'andamento della trasparenza dei cristalli ECAL

# Importo le librerie necessarie
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#importo il file csv e creo il dataframe dati
df_dati= pd.read_csv(r"/home/marta/python/fill_metadata_2017_10min.csv", usecols = [0,1,7])
#escludo dal dataframe le linee con in_fill=0 e quelle che non corrispondono al fill scelto
df_dati=df_dati.drop(df_dati[df_dati.in_fill==0].index)
fill = 5872
df_dati=df_dati.drop(df_dati[df_dati.fill_num!=fill].index)
#converto il tempo dal formato timestamp a giorni
df_dati.time=(df_dati.time/86400)-17309
#from datetime import datetime
#df_dati=df_dati.assign(data = 0)
#df_dati.data = datetime.fromtimestamp(df_dati.time)

# Creo un dataframe con i valori della trasparenza
#importo il file npy e creo il dataframe trasparenza
trasparenza = np.load(r"/home/marta/python/2-TraspVsTime/iRing25new.npy")
df_trasparenza = pd.DataFrame(trasparenza)
#inverto righe e colonne del dataframe sulla trasparenza
df_trasparenza= df_trasparenza.transpose()
#aggiungo una colonna al dataframe trasparenza, che poi conterra la media delle trasparenze
df_trasparenza=df_trasparenza.assign(trasparenza = 0)
#faccio la somma per riga e divido per il numero di colonne per ottenere la media
df_trasparenza.trasparenza=df_trasparenza.sum(axis=1)/311
#escludo dal dataframe le linee con trasparenza < 0
df_trasparenza=df_trasparenza.drop(df_trasparenza[df_trasparenza.trasparenza<0].index)
#normalizzo la trasparenza dividendo per la traspareza della riga 0 (0.596746)
df_trasparenza.trasparenza=df_trasparenza.trasparenza/0.596746

#aggiungo la colonna trasparenza al dataframe dati
df_dati=df_dati.assign(trasparenza = 0)
df_dati.trasparenza=df_trasparenza.trasparenza
df_dati=df_dati.drop(df_dati[df_dati.trasparenza<0].index)
#tolgo una colonna (in_fill) al dataframe dati
df_dati.drop(df_dati.columns[[0]], axis = 1, inplace = True)

# stampo il df dati
print(df_dati)
#print(type(df_dati))

#faccio il plot trasparenza vs indice
#df_dati.plot(kind='scatter', x='time', y='trasparenza')
#plt.xlabel("Time  [days]")
#plt.ylabel("Normalized mean transparency")
#plt.title("Normalized mean transparency vs time")
#plt.show()

#nuovo dataframe solo con la trasparenza
data = df_dati.filter(['trasparenza'])
#Convertiamolo in un numpy array
dataset = data.values
training_data_len = math.ceil( len(dataset) *.8)
#Crea il set di dati per l'addestramento in scala
train_data = dataset[0:training_data_len  , : ]
#Dividi i dati nel x_train e  y_train
x_train=[]
y_train = []
for i in range(60,len(train_data)):
 x_train.append(train_data[i-60:i,0])
 y_train.append(train_data[i,0])
 # Converti il ​​set di dati indipendente " x_train " e il set di dati  dipendente " y_train " in array numpy 
# in modo che possano essere utilizzati per l'addestramento del modello LSTM.
x_train, y_train = np.array(x_train), np.array(y_train)

# Il modello LSTM si aspetta un set di dati tridimensionale.
# Reshape i dati nella forma accettata da LSTM
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# Costruisci il modello LSTM per avere due strati LSTM con 50 neuroni e due strati densi, uno con 25 neuroni 
# e l'altro con 1 neurone di output.
# Costruiamo il LSTM network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

#Compila il modello model.compile. utilizzando la funzione di perdita dell'errore quadratico medio (MSE) 
# e l'ottimizzatore adam.
model.compile(optimizer='adam', loss='mean_squared_error')

#Allena il modello. La dimensione del batch è il numero totale di esempi di addestramento presenti in un 
# singolo batch ed epoch è il numero di iterazioni in cui un intero set di dati viene passato avanti e 
# indietro attraverso la rete neurale.
model.fit(x_train, y_train, batch_size=1, epochs=10)

#Crea un set di dati di test.
test_data = dataset[training_data_len - 60: , : ]
#Crea i set di dati x_test e y_test
x_test = []
y_test = dataset[training_data_len : , : ] 
#Recupera tutte le righe dall'indice 1603 al resto e tutte le colonne (in questo caso è solo la colonna "Chiudi"), 
# così 2003 - 1603 = 400 righe di dati . Commento copiato dal web, non sono riuscita a capire cosa significhi
for i in range(60,len(test_data)):
 x_test.append(test_data[i-60:i,0])

# Converti x_test in un array numpy 
x_test = np.array(x_test)

#Reshape i dati nella forma accettata da LSTM
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

#Otteniamo le predizioni del modello
predictions = model.predict(x_test) 

# Ottieni l'errore quadratico medio (RMSE), che è una buona misura dell'accuratezza del modello. 
# Un valore pari a 0 indica che i valori previsti dai modelli corrispondono perfettamente ai valori 
# effettivi del set di dati del test. Più basso è il valore, migliori saranno le prestazioni del modello. 
# Ma di solito è meglio usare anche altre metriche per avere davvero un'idea di come si è comportato bene il modello.

# Calcola / Ottieni il valore di RMSE
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
print(rmse)

# Crea i dati per il grafico e converto in un dataframe
train = data[:training_data_len]
train.columns = ['Train']   #rinomina la colonna

valid = data[training_data_len:]
valid.columns = ['Valid']   #rinomina la colonna

valid['Predictions'] = predictions

# unisco i dataframe train e valid
df = valid.merge(train, how='outer', left_index=True, right_index=True)
df=df.assign(Time = 0)
df.Time = df_dati.time

#from datetime import datetime
#now = datetime.now()
#df=df.assign(date = now)
#df.date = datetime.fromtimestamp(df.time)
#print("df")
print(df)

#faccio il grafico
#plt.figure(figsize=(16,8))
df.plot(x="Time", y=["Train", "Valid", "Predictions"])
plt.xlabel("Time [days]")
plt.ylabel("Normalized mean transparency")
plt.title(f"Normalized mean transparency: data and predictions - fill {fill}")
plt.show()

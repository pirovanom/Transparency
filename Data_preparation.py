# DESCRIZIONE: Preparo il file csv di input per il modello LSTM

# importo le librerie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importo il file csv e creo il dataframe df_dati
df_dati= pd.read_csv(r"/home/marta/python/fill_metadata_2017_10min.csv", usecols = [0,1,2,3,4,5,6,7,8])
# Escludo dal dataframe le linee con in_fill=0 (e se decommento due righe quelle che non corrispondono al fill scelto)
df_dati=df_dati.drop(df_dati[df_dati.in_fill==0].index)
#fill = 5872
#df_dati=df_dati.drop(df_dati[df_dati.fill_num!=fill].index)
# Escludo i dati dei fill non buoni
escludi = list()
escludi = [5697, 5698, 5699, 5704, 5710, 5718, 5719, 5722, 5733, 5736, 5738, 5739, 5740, 5746, 5749, 5822, 5824, 5825,
           5830, 5833, 5834, 5837, 5838, 5859, 5860, 5861, 5862, 5868, 5870, 5871, 5874, 5874, 5885, 5919, 5920, 5929, 
           5933, 5946, 5952, 5954, 5960, 5965, 5970, 5971, 5974, 5985, 6001, 6005, 6006, 6012, 6015, 6016, 6018, 6019, 
           6021, 6034, 6036, 6039, 6041, 6047, 6055, 6057, 6072, 6123, 6130, 6132, 6141, 6146, 6155, 6164, 6173, 6179, 
           6180, 6183, 6184, 6185, 6192, 6194, 6195, 6199, 6200, 6201, 6216, 6217, 6226, 6227, 6228, 6230, 6231, 6232, 
           6233, 6236, 6238, 6261, 6293, 6294, 6295, 6309, 6313, 6318, 6336, 6337, 6341, 6344, 6347, 6348, 6349, 6351, 
           6355, 6358, 6370, 6374, 6377, 6380, 6381, 6382, 6385, 6387, 6388, 6396, 6399, 6402, 6404, 6405, 6411, 6413, 
           6415, 6417, 6431, 6434]
for k in escludi:
    df_dati=df_dati.drop(df_dati[df_dati.fill_num==k].index)

# Creo un dataframe con i valori della trasparenza

# Importo il file npy e creo il dataframe df_trasparenza
trasparenza = np.load(r"/home/marta/python/2-TraspVsTime/iRing25new.npy")
df_trasparenza = pd.DataFrame(trasparenza)
# Inverto righe e colonne del dataframe df_trasparenza
df_trasparenza= df_trasparenza.transpose()
# Aggiungo una colonna al dataframe df_trasparenza, che poi conterra la media delle trasparenze
df_trasparenza=df_trasparenza.assign(trasparenza = 0)
# Faccio la somma per riga e divido per il numero di colonne per ottenere la media
df_trasparenza.trasparenza=df_trasparenza.sum(axis=1)/311
# Escludo dal dataframe le linee con trasparenza < 0
df_trasparenza=df_trasparenza.drop(df_trasparenza[df_trasparenza.trasparenza<0].index)

# Aggiungo la colonna trasparenza al dataframe dati
df_dati=df_dati.assign(trasparenza = 0)
df_dati.trasparenza=df_trasparenza.trasparenza
df_dati=df_dati.drop(df_dati[df_dati.trasparenza<0].index)
# Tolgo una colonna (in_fill) al dataframe dati
df_dati.drop(df_dati.columns[[0]], axis = 1, inplace = True)
# Sostituisco con 0 i valori NA
df_dati['trasparenza'].fillna(0, inplace=True)
# Escludo dal dataframe le linee con trasparenza 0
df_dati.drop(df_dati.loc[df_dati['trasparenza']==0.0].index, inplace=True)

# Normalizzo la trasparenza dividendo per il massimo valore di traspareza per ogni fill 

# Creo una lista contente tutti i numeri di fill presenti nel dataframe df_dati
fill_nums = df_dati.fill_num.unique()
# Creo un dataframe vuoto
dataframe = pd.DataFrame()
# Per ogni fill scalo i valori di trasparenza e aggiungo il nuovo dataframe al dataframe vuoto appena creato
for k in (fill_nums):
    new_df = df_dati[df_dati.fill_num == k]
    massimo_trasp = new_df.trasparenza.iloc[0]
    minimo_trasp = new_df['trasparenza'].min()
    new_df.trasparenza = new_df.trasparenza / massimo_trasp
    dataframe = dataframe.append(new_df, ignore_index=True)

# Stampo i dataframe 
#print("df_dati")
#print(df_dati)
print("dataframe")
print(dataframe)

# Salvo il dataframe in un file csv
dataframe.to_csv('/home/marta/python/5-Correlazioni/correlazioni.csv')

# Scalo a mano tutti i dati tranne la trasparenza, per portarli nel range (0, 1) perchè così li vuole la rete neurale
massimo = dataframe['time_in_fill'].max()
minimo = dataframe['time_in_fill'].min()
dataframe.time_in_fill = (dataframe.time_in_fill - minimo) / (massimo - minimo)

massimo = dataframe['lumi_last_fill'].max()
minimo = dataframe['lumi_last_fill'].min()
dataframe.lumi_last_fill = (dataframe.lumi_last_fill - minimo) / (massimo - minimo)

massimo = dataframe['lumi_since_last_point'].max()
minimo = dataframe['lumi_since_last_point'].min()
dataframe.lumi_since_last_point = (dataframe.lumi_since_last_point - minimo) / (massimo - minimo)

massimo = dataframe['lumi_in_fill'].max()
minimo = dataframe['lumi_in_fill'].min()
dataframe.lumi_in_fill = (dataframe.lumi_in_fill - minimo) / (massimo - minimo)

massimo = dataframe['lumi_int'].max()
minimo = dataframe['lumi_int'].min()
dataframe.lumi_int = (dataframe.lumi_int - minimo) / (massimo - minimo)

massimo = dataframe['lumi_inst'].max()
minimo = dataframe['lumi_inst'].min()
dataframe.lumi_inst = (dataframe.lumi_inst - minimo) / (massimo - minimo)

plt.scatter(dataframe.time, dataframe.trasparenza)
plt.title("Trasparenza")
plt.show()

"""plt.plot(dataframe.time_in_fill)
plt.title("time_in_fill")
plt.show()

plt.plot(dataframe.fill_num)
plt.title("fill_num")
plt.show()

plt.plot(dataframe.lumi_last_fill)
plt.title("lumi_last_fill")
plt.show()

plt.plot(dataframe.lumi_since_last_point)
plt.title("lumi_since_last_point")
plt.show()

plt.plot(dataframe.lumi_in_fill)
plt.title("lumi_in_fill")
plt.show()

plt.plot(dataframe.lumi_int)
plt.title("lumi_int")
plt.show()

plt.plot(dataframe.lumi_inst)
plt.title("lumi_inst")
plt.show()"""

# Creo un file csv con solo i dati del tempo

time = dataframe.copy()
time.drop(['lumi_inst'], axis = 1, inplace = True)
time.drop(['lumi_int'], axis = 1, inplace = True)
time.drop(['time_in_fill'], axis = 1, inplace = True)
time.drop(['lumi_since_last_point'], axis = 1, inplace = True)
time.drop(['fill_num'], axis = 1, inplace = True)
time.drop(['lumi_in_fill'], axis = 1, inplace = True)
time.drop(['lumi_last_fill'], axis = 1, inplace = True)
time.drop(['trasparenza'], axis = 1, inplace = True)
#print("time")
#print(time)
time.set_index('time', drop = False, inplace = True)
#time.time = [datetime.datetime.fromtimestamp(ts) for ts in time.time]
#time.index = [datetime.datetime.fromtimestamp(ts) for ts in time.index]
print("time")
print(time)
time.to_csv('/home/marta/python/7-LSTM_prova2/time.csv')

# Faccio un file csv con solo i dati del numero di fill

fill_n = dataframe.copy()
fill_n.drop(['lumi_inst'], axis = 1, inplace = True)
fill_n.drop(['lumi_int'], axis = 1, inplace = True)
fill_n.drop(['time_in_fill'], axis = 1, inplace = True)
fill_n.drop(['lumi_since_last_point'], axis = 1, inplace = True)
fill_n.drop(['lumi_in_fill'], axis = 1, inplace = True)
fill_n.drop(['lumi_last_fill'], axis = 1, inplace = True)
fill_n.drop(['trasparenza'], axis = 1, inplace = True)
fill_n.set_index('time', drop = True, inplace = True)
print("fill_n")
print(fill_n)
fill_n.to_csv('/home/marta/python/7-LSTM_prova2/fill_n.csv')

# Definisco la colonna time come indice

dataframe.set_index('time', drop = True, inplace = True)
print(dataframe)
plt.plot(dataframe)
plt.show()

# Salvo i dati in un file csv, che userò come input per la rete neurale LSTM
dataframe.to_csv('/home/marta/python/7-LSTM_prova2/input_data.csv')



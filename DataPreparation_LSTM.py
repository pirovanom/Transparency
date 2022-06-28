# Preparo il file csv di input per il modello

import pandas as pd
import numpy as np

#importo il file csv e creo il dataframe dati
df_dati= pd.read_csv(r"/home/marta/python/fill_metadata_2017_10min.csv", usecols = [0,1,2,3,4,5,6,7,8])
#escludo dal dataframe le linee con in_fill=0 e quelle che non corrispondono al fill scelto
df_dati=df_dati.drop(df_dati[df_dati.in_fill==0].index)
#fill = 5872
#df_dati=df_dati.drop(df_dati[df_dati.fill_num!=fill].index)
#converto il tempo dal formato timestamp a giorni
df_dati.time=(df_dati.time/86400)-17309
#escludo i dati dei fill non buoni
escludi = list()
escludi = [5697, 5698, 5699, 5704, 5710, 5718, 5719, 5722, 5733, 5736, 5738, 5739, 5740, 5746, 5749, 5822, 5824, 5825,
           5830, 5833, 5834, 5837, 5838, 5859, 5860, 5861, 5862, 5868, 5870, 5871, 5874, 5874, 5885, 5919, 5920, 5929, 
           5933, 5946, 5952, 5954, 5960, 5965, 5970, 5971, 5974, 5985, 6001, 6005, 6006, 6012, 6015, 6016, 6018, 6019, 
           6021, 6034, 6036, 6039, 6041, 6047, 6055, 6057, 6072, 6123, 6130, 6132, 6141, 6146, 6155, 6164, 6173, 6179, 
           6180, 6183, 6184, 6185, 6192, 6194, 6195, 6199, 6200, 6201, 6216, 6217, 6226, 6227, 6228, 6230, 6231, 6232, 
           6233, 6236, 6238, 6261, 6293, 6294, 6295, 6309, 6336, 6341, 6351, 6374, 6377, 6380, 6381, 6382, 6387, 6388, 
           6399, 6402, 6404, 6405, 6411, 6413, 6415, 6417, 6431]
for k in escludi:
    df_dati=df_dati.drop(df_dati[df_dati.fill_num==k].index)

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

#aggiungo la colonna trasparenza al dataframe dati
df_dati=df_dati.assign(trasparenza = 0)
df_dati.trasparenza=df_trasparenza.trasparenza
df_dati=df_dati.drop(df_dati[df_dati.trasparenza<0].index)
#tolgo una colonna (in_fill) al dataframe dati
df_dati.drop(df_dati.columns[[0]], axis = 1, inplace = True)
#for i in range (1, len(df_dati.index)):
    #valore = df_dati.at[i-1, 'trasparenza']
    #print(df_dati.at[i, 'trasparenza'])
    #df_dati.at[i, 'trasp_preced'] = valore
#sostituisco con 0 i valori NA
df_dati['trasparenza'].fillna(0, inplace=True)
#escludo dal dataframe le linee con trasparenza 0
df_dati.drop(df_dati.loc[df_dati['trasparenza']==0.0].index, inplace=True)
#normalizzo la trasparenza dividendo per il massimo valore di traspareza 
massimo_trasp = df_dati['trasparenza'].max()
df_dati.trasparenza = df_dati.trasparenza/massimo_trasp
#aggiungo una colonna trasparenza shiftata di 1
#df_dati=df_dati.assign(trasp_preced = 0)
#df_dati.trasp_preced = df_dati.trasparenza

#faccio un file csv con solo i dati del tempo
time = df_dati.copy()
#time.set_index('indice', drop = False, inplace = True)
time.drop(['lumi_inst'], axis = 1, inplace = True)
time.drop(['lumi_int'], axis = 1, inplace = True)
time.drop(['time_in_fill'], axis = 1, inplace = True)
time.drop(['lumi_since_last_point'], axis = 1, inplace = True)
time.drop(['fill_num'], axis = 1, inplace = True)
time.drop(['lumi_in_fill'], axis = 1, inplace = True)
time.drop(['lumi_last_fill'], axis = 1, inplace = True)
time.drop(['trasparenza'], axis = 1, inplace = True)
#time.drop(['trasp_preced'], axis = 1, inplace = True)
print("time")
print(time)
time.to_csv('/home/marta/python/7-LSTM_prova2/time.csv')

#definisco la colonna time come indice
df_dati.set_index('time', drop = True, inplace = True)

# stampo il df dati
print("df_dati")
print(df_dati)
#print(type(df_dati))
fill_num = df_dati.fill_num.unique()
"""i=0
for k in fill_num:
    print(f"{i}, {k}")
    i=i+1"""

# salvo i dati in un file csv
df_dati.to_csv('/home/marta/python/7-LSTM_prova2/input_data.csv')



import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#importo il file csv e creo il dataframe time
df_time= pd.read_csv(r"/home/marta/python/fill_metadata_2017_10min.csv", usecols = [0,1,2,3,4,7])
#escludo dal dataframe le linee con in_fill=0
df_time=df_time.drop(df_time[df_time.in_fill==0].index)
#converto il tempo dal formato timestamp a giorni
df_time.time=(df_time.time/86400)-17309

#importo il file npy e creo il dataframe trasparenza
trasparenza = np.load(r"/home/marta/python/2-TraspVsTime/iRing23new.npy")
df_trasparenza = pd.DataFrame(trasparenza)
#inverto righe e colonne del dataframe sulla trasparenza
df_trasparenza= df_trasparenza.transpose()
#aggiungo una colonna al dataframe trasparenza, che poi conterra la media delle trasparenze
df_trasparenza=df_trasparenza.assign(trasparenza = 0)
#faccio la somma per riga e divido per il numero di colonne per ottenere la media
df_trasparenza.trasparenza=df_trasparenza.sum(axis=1)/311
#escludo dal dataframe le linee con trasparenza < 0
df_trasparenza=df_trasparenza.drop(df_trasparenza[df_trasparenza.trasparenza<0].index)
#normalizzo la trasparenza dividendo per la traspareza della rigo 0 (0,720461)
df_trasparenza.trasparenza=df_trasparenza.trasparenza/0.720461

#unisco i due dataframe e stampo il df finale
df_unico = df_time.merge(df_trasparenza, how='inner', left_index=True, right_index=True)
print(df_unico)
print(type(df_unico))

#faccio il plot
df_unico.plot(kind='scatter', x='time', y='trasparenza')
plt.xlabel("Time  [days]")
plt.ylabel("Normalized mean transparency")
plt.title("Normalized mean transparency vs time")
plt.show()

#faccio il plot per il singolo fill
fill_num = df_unico.fill_num.unique()
for k in fill_num:
    df = df_unico[df_unico.fill_num == k]
    df.plot(kind='line', x='time', y='trasparenza')
    plt.xlabel("Time  [days]")
    plt.ylabel("Normalized mean transparency")
    plt.title(f"Fill numero {k}")
    plt.show()

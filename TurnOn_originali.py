# DESCRIZIONE: questo codice fa il plot del numero di eventi salvati dal trigger di CMS per ogni valore di energia. 
# Questo tipo di plot è chiamato Turn On Curves

# Importo le librerie
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Creo un dataframe con i dati di output del modello LSTM
df = pd.read_csv('/home/marta/python/7-LSTM_prova2/output.csv', header=0, index_col=0)
#print(df)

# Creo dei vettori con i dati di test e con le predizioni e li trasformo in numpy array
test = df['test']
predictions = df['predictions']

test = test.to_numpy()
predictions = predictions.to_numpy()

# Definisco alcuni parametri
e_max=32                                    # energia massima considerata
e_min=28                                    # energia minima considerata
num_eventi = len(predictions)
#print("numero totale di eventi:  ", num_eventi)
bin_desiderati = 200                        # numero di interavalli (bin) in cui dividere il range di energia considerata
larg=(e_max-e_min)/bin_desiderati           # largezza di ogni bin
num_bin = int((e_max - e_min)/ larg)        # numero di bin (per controllo)
soglia = 30                                 # soglia di energia in GeV
#print("num bin: ", num_bin)

# Inizializzo alcune quantità
k=0
cont = 0
energia = list()

# Creo il vettore con i valori di energia
for i in range(0, num_bin):
    energia.append(e_min + (larg*(k)))
    k = k+1
#print(energia)

# TUTTI I FILL

# Riempio il vettore conteggi_test con i conteggi che si ottengono NON CONSIDERANDO le correzioni alla trasparenza 
# ottenute con LSTM
cont_test = list()
for i in range(0, num_bin):
    for j in range (0, len(test)):
        en_corr = energia[i] * test[j]
        if (en_corr > soglia):
            cont = cont + 1
    #print(cont)
    cont_test.append(cont) 
    cont = 0

# Riempio il vettore conteggi_pred con i conteggi che si ottengono CONSIDERANDO le correzioni alla trasparenza ottenute
# con LSTM
cont_pred = list()
for i in range(0, num_bin):
    for j in range (0, len(predictions)):
        en_corr = energia[i] * test[j] / predictions[j]
        if (en_corr > soglia):
            cont = cont + 1
    #print(cont)
    cont_pred.append(cont) 
    cont = 0

# Creo un dataframe con l'energia e i conteggi
df_result = pd.DataFrame(energia)
df_result=df_result.assign(cont_pred = 0)
df_result.cont_pred = pd.DataFrame(cont_pred)
df_result=df_result.assign(cont_test = 0)
df_result.cont_test = pd.DataFrame(cont_test)
print(df_result)

# Scalo i conteggi per portarli tra 0 e 1
df_result.cont_pred = df_result.cont_pred / num_eventi
df_result.cont_test = df_result.cont_test / num_eventi
# Rinomino le colonne del daataframe per renderlo più comprensibile
df_result.set_axis(['energia', 'cont_pred', 'cont_test'], axis='columns', inplace=True)
df_result.to_csv('/home/marta/python/10-TurnOn/Originali/turn_on_originali.csv')
#print(df)

# Plot delle turn on curve per tutti i fill
plt.plot(df_result.energia, df_result.cont_pred, ".r-", label = "Predictions")
plt.plot(df_result.energia, df_result.cont_test, ".y-", label = "Test")
plt.xlabel("Energy [GeV]")
plt.ylabel(f"Fraction of events with measured energy greater than {soglia} GeV")
plt.title(f"Trigger efficiency - All fill")
legend = plt.legend(['CM4S measure with predictions correction', 'CMS measure without predictions correction'], title = "Legend", loc="center left")
plt.savefig("/home/marta/python/10-TurnOn/Originali/TurnOn_originali_TuttiIFill.pdf")
plt.show()


#SINGOLI FILL: analogo a sopra ma con i dati di singoli fill 

# Definisco dei nuovi dataframe contenenti porzioni dell'originale, divisi in base al numero del fill
#print(df)
fill_nums = df.fill_num.unique()
for f in (fill_nums):
    new_df = df[df.fill_num == f]
    #definisco i nuovi vettori test e predictions
    test = new_df['test']
    predictions = new_df['predictions']
    test = test.to_numpy()
    predictions = predictions.to_numpy()
    num_eventi = len(predictions)
    print(num_eventi)
    # Riempio il vettore conteggi_test NON CONDSIDERANDO le correzioni alla trasparenza LSTM
    cont_test = list()
    for i in range(0, num_bin):
        for j in range (0, len(test)):
            en_corr = energia[i] * test[j]
            if (en_corr > soglia):
                cont = cont + 1
        #print(cont)
        cont_test.append(cont) 
        cont = 0
    # Riempio il vettore conteggi_pred CONDSIDERANDO le correzioni alla trasparenza LSTM
    cont_pred = list()
    for i in range(0, num_bin):
        for j in range (0, len(predictions)):
            en_corr = energia[i] * test[j] / predictions[j]
            if (en_corr > soglia):
                cont = cont + 1
        #print(cont)
        cont_pred.append(cont) 
        cont = 0
    # Creo un dataframe con l'energia e i conteggi
    df_result = pd.DataFrame(energia)
    df_result=df_result.assign(cont_pred = 0)
    df_result.cont_pred = pd.DataFrame(cont_pred)
    df_result=df_result.assign(cont_test = 0)
    df_result.cont_test = pd.DataFrame(cont_test)
    #print(df_result)
    # Scalo i conteggi per portarli tra 0 e 1
    df_result.cont_pred = df_result.cont_pred / num_eventi
    df_result.cont_test = df_result.cont_test / num_eventi
    # Rinomino le colonne del daataframe per renderlo più comprensibile
    df_result.set_axis(['energia', 'cont_pred', 'cont_test'], axis='columns', inplace=True)
    #print(df)
    # Plot delle turn on curve per singoli fill
    plt.plot(df_result.energia, df_result.cont_pred, ".r-", label = "Predictions")
    plt.plot(df_result.energia, df_result.cont_test, ".y-", label = "Test")
    plt.xlabel("Energy [GeV]")
    plt.ylabel(f"Fraction of events with measured energy greater than {soglia} GeV")
    plt.title(f"Trigger efficiency - Fill {f}")
    legend = plt.legend(['CM4S measure with predictions correction', 'CMS measure without predictions correction'], title = "Legend", loc="center left")
    plt.savefig(f"/home/marta/python/10-TurnOn/Originali/PerFill/TurnOn_Fill_{f}.pdf")
    plt.show()

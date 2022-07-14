#Turn On Curves pesate per la luminosità

#importo le librerie
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import seaborn as sns 
import matplotlib.dates as mdates

# creo un dataframe con i dati di output del modello LSTM
df = pd.read_csv('/home/marta/python/7-LSTM_prova2/output.csv', header=0, index_col=0)
print(df)

# creo dei vettori con i dati di test e con le predizioni e li trasformo in numpy array
test = df['test']
predictions = df['predictions']
lumi_inst = df['lumi_inst']

test = test.to_numpy()
predictions = predictions.to_numpy()
lumi_inst = lumi_inst.to_numpy()

#definisco i parametri importanti
e_max=32
e_min=28
#num_eventi = len(predictions)
#print("numero totale di eventi:  ", num_eventi)
bin_desiderati = 200
larg=(e_max-e_min)/bin_desiderati
num_bin = int((e_max - e_min)/ larg)
soglia = 30
print("num bin: ", num_bin)

print("len(lumi_inst)", len(lumi_inst))
somma_lumi = 0
for m in range (len(lumi_inst)):
    somma_lumi = somma_lumi + lumi_inst[m]
print("somma totale delle luminosità:  ", somma_lumi)


#inizializzo alcune quantità
k=0
cont = 0
energia = list()

# creo il vettore con i valori di energia
for i in range(0, num_bin):
    energia.append(e_min + (larg*(k)))
    k = k+1
#print(energia)

# TUTTI I FILL

# riempio il vettore conteggi_test con i conteggi a partire dalla trasparenza di test
cont_test = list()
for i in range(0, num_bin):
    for j in range (0, len(test)):
        en_corr = energia[i] * test[j]
        if (en_corr > soglia):
            cont = cont + lumi_inst[j]
    #print(cont)
    cont_test.append(cont) 
    cont = 0

# riempio il vettore conteggi_pred con i conteggi a partire dalla trasparenza predetta
cont_pred = list()
for i in range(0, num_bin):
    for j in range (0, len(predictions)):
        en_corr = energia[i] * predictions[j]
        if (en_corr > soglia):
            cont = cont + lumi_inst[j]
    #print(cont)
    cont_pred.append(cont) 
    cont = 0

#creo un dataframe con l'energia e i conteggi
df_result = pd.DataFrame(energia)
df_result=df_result.assign(cont_pred = 0)
df_result.cont_pred = pd.DataFrame(cont_pred)
df_result=df_result.assign(cont_test = 0)
df_result.cont_test = pd.DataFrame(cont_test)
print(df_result)

# scalo i conteggi per portarli tra 0 e 1
df_result.cont_pred = df_result.cont_pred / somma_lumi
df_result.cont_test = df_result.cont_test / somma_lumi
# rinomino le colonne del daataframe per renderlo più comprensibile
df_result.set_axis(['energia', 'cont_pred', 'cont_test'], axis='columns', inplace=True)
df_result.to_csv('/home/marta/python/10-TurnOn/Turn_On_pesate_lumi_inst/turn_on_lumi.csv')
#print(df)

#plot delle turn on curve per tutti i fill
plt.plot(df_result.energia, df_result.cont_pred, ".r-", label = "Predictions")
plt.plot(df_result.energia, df_result.cont_test, ".y-", label = "Test")
#plt.plot(new_df.time, new_df.predictions, ".b-", markersize=3, linewidth=0.75, label="Predictions")
plt.xlabel("Energy [GeV]")
plt.ylabel(f"Fraction of events with measured energy greater than {soglia} GeV")
plt.title(f"Trigger efficiency - All fill")
legend = plt.legend(['Predictions', 'Test'], title = "Legend")
plt.savefig("/home/marta/python/10-TurnOn/Turn_On_pesate_lumi_inst/TurnOn_lumi_TuttiIFill.pdf")
plt.show()


#SINGOLI FILL
#definisco dei nuovi dataframe contenenti porzioni dell'originale, diviso per nuero di fill
print("")
print("SINGLE FILLS")
df_fills =pd.DataFrame()
fill_nums = df.fill_num.unique()
for f in (fill_nums):
    print("")
    new_df = df[df.fill_num == f]
    #definisco i nuovi vettori test e predictions
    test = new_df['test']
    predictions = new_df['predictions']
    lumi_inst = new_df['lumi_inst']
    test = test.to_numpy()
    predictions = predictions.to_numpy()
    lumi_inst = lumi_inst.to_numpy()
    print("len(lumi_inst)", len(lumi_inst))
    print("len(predict)", len(predictions))
    somma_lumi = 0
    for m in range (len(lumi_inst)):
        somma_lumi = somma_lumi + lumi_inst[m]
    print("somma totale delle luminosità:  ", somma_lumi)
    #num_eventi = len(predictions)
    #print(num_eventi)
    # riempio il vettore conteggi_test con i conteggi a partire dalla trasparenza di test
    cont_test = list()
    for i in range(0, num_bin):
        for j in range (0, len(test)):
            en_corr = energia[i] * test[j]
            if (en_corr > soglia):
                cont = cont + lumi_inst[j]
        #print(cont)
        cont_test.append(cont) 
        cont = 0
    # riempio il vettore conteggi_pred con i conteggi a partire dalla trasparenza predetta
    cont_pred = list()
    for i in range(0, num_bin):
        for j in range (0, len(predictions)):
            en_corr = energia[i] * predictions[j]
            if (en_corr > soglia):
                cont = cont + lumi_inst[j]
        #print(cont)
        cont_pred.append(cont) 
        cont = 0
    #creo un dataframe con l'energia e i conteggi
    df_result = pd.DataFrame(energia)
    df_result=df_result.assign(cont_pred = 0)
    df_result.cont_pred = pd.DataFrame(cont_pred)
    df_result=df_result.assign(cont_test = 0)
    df_result.cont_test = pd.DataFrame(cont_test)
    #print(df_result)
    # scalo i conteggi per portarli tra 0 e 1
    df_result.cont_pred = df_result.cont_pred / somma_lumi
    df_result.cont_test = df_result.cont_test / somma_lumi
    # rinomino le colonne del daataframe per renderlo più comprensibile
    df_result.set_axis(['energia', 'cont_pred', 'cont_test'], axis='columns', inplace=True)
    df_result=df_result.assign(fill_num = f)
    #print(df)
    #plot delle turn on curve per tutti i fill
    plt.plot(df_result.energia, df_result.cont_pred, ".r-", label = "Predictions")
    plt.plot(df_result.energia, df_result.cont_test, ".y-", label = "Test")
    plt.xlabel("Energy [GeV]")
    plt.ylabel(f"Fraction of events with measured energy greater than {soglia} GeV")
    plt.title(f"Trigger efficiency - Fill {f}")
    legend = plt.legend(['Predictions', 'Test'], title = "Legend")
    plt.savefig(f"/home/marta/python/10-TurnOn/Turn_On_pesate_lumi_inst/PerFill/TurnOn_lumi_Fill_{f}.pdf")
    df_fills = df_fills.append(df_result, ignore_index=True)
    plt.show()
"""print("df_fills")
print(df_fills)
print("df")
print(df)"""

# DESCRIZIONE: questo codice fa il plot del numero di eventi salvati dal trigger di CMS per ogni valore di energia, 
# pesandoli per la luminosità istantanea in LHC. Questo tipo di plot è chiamato Turn On Curves pesate per la luminosità

# Importo le librerie
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Creo un dataframe con i dati di output del modello LSTM
df = pd.read_csv('/home/marta/python/7-LSTM_prova2/output.csv', header=0, index_col=0)
print(df)

# Creo dei vettori con i dati di test, di luminosità istantanea e con le predizioni e li trasformo in numpy array
test = df['test']
predictions = df['predictions']
lumi_inst = df['lumi_inst']

test = test.to_numpy()
predictions = predictions.to_numpy()
lumi_inst = lumi_inst.to_numpy()

# Definisco alcuni parametri 
e_max=32                                # energia massima considerata
e_min=28                                # energia minima considerata
bin_desiderati = 200                    # numero di interavalli (bin) in cui dividere il range di energia considerata
larg=(e_max-e_min)/bin_desiderati       # largezza di ogni bin
num_bin = int((e_max - e_min)/ larg)    # numero di bin (per controllo)
soglia = 30                             # soglia di energia in GeV
print("num bin: ", num_bin)

somma_lumi = 0                          # inizializzo a zero
for m in range (len(lumi_inst)):        # sommo tutte le luminosità istantanee per tutti i fill
    somma_lumi = somma_lumi + lumi_inst[m]
print("somma totale delle luminosità:  ", somma_lumi)

# Inizializzo alcune quantità
k=0
cont = 0
energia = list()
senza  = 0
con = 0
"""print(len(energia))
print(len(test))"""

# Creo il vettore con i valori di energia
for i in range(0, num_bin):
    energia.append(e_min + (larg*(k)))
    k = k+1
#print(energia)

# TUTTI I FILL
eventi_tot = len(energia)*len(test)

# Riempio il vettore conteggi_test con i conteggi che si ottengono NON CONSIDERANDO le correzioni alla trasparenza 
# ottenute con LSTM
cont_misura_attuale = list()
for i in range(0, num_bin):
    for j in range (0, len(test)):
        en_corr = energia[i] * test[j]
        if (en_corr > soglia):
            senza = senza + 1
            cont = cont + lumi_inst[j]
    #print(cont)
    cont_misura_attuale.append(cont) 
    cont = 0

# Riempio il vettore conteggi_pred con i conteggi che si ottengono CONSIDERANDO le correzioni alla trasparenza ottenute
# con LSTM
cont_corretto = list()
for i in range(0, num_bin):
    for j in range (0, len(predictions)):
        en_corr = energia[i] * test[j] / predictions[j]
        if (en_corr > soglia):
            con = con + 1
            cont = cont + lumi_inst[j]
    #print(cont)
    cont_corretto.append(cont) 
    cont = 0

# Creo un dataframe con l'energia e i conteggi nei due casi
df_result = pd.DataFrame(energia)
df_result=df_result.assign(cont_corretto = 0)
df_result.cont_corretto = pd.DataFrame(cont_corretto)
df_result=df_result.assign(cont_misura_attuale = 0)
df_result.cont_misura_attuale = pd.DataFrame(cont_misura_attuale)
print(df_result)

# Scalo i conteggi per portarli tra 0 e 1
df_result.cont_corretto = df_result.cont_corretto / somma_lumi
df_result.cont_misura_attuale = df_result.cont_misura_attuale / somma_lumi
# Rinomino le colonne del dataframe per renderlo più comprensibile
df_result.set_axis(['energia', 'cont_corretto', 'cont_misura_attuale'], axis='columns', inplace=True)
# Salvo il dataframe in formato csv
df_result.to_csv('/home/marta/python/10-TurnOn/Turn_On_pesate_lumi_inst/turn_on_lumi.csv')
#print(df)

# Plot delle turn on curve per tutti i fill
plt.plot(df_result.energia, df_result.cont_corretto, ".b-", label = "Predictions")
plt.plot(df_result.energia, df_result.cont_misura_attuale, ".g-", label = "Test")
plt.xlabel("Energy [GeV]")
plt.ylabel(f"Fraction of events saved per instantaneous luminosity unit")
plt.title(f"Trigger efficiency - All fill")
legend = plt.legend(['CMS measure with predictions correction', 'CMS measure without predictions correction'], title = "Legend", loc="center left")
plt.savefig("/home/marta/python/10-TurnOn/Turn_On_pesate_lumi_inst/TurnOn_lumi_TuttiIFill.pdf")
plt.show()

"""print("con:  ", con)
print("senza:  ", senza)
print("Eventi_tot:  ", eventi_tot)
print("Percentuale:  ", (con-senza)/eventi_tot)"""

#SINGOLI FILL

# Definisco dei nuovi dataframe contenenti porzioni dell'originale, divisi in base al numero del fill
print("")
print("SINGLE FILLS")
df_fills =pd.DataFrame()
fill_nums = df.fill_num.unique()
for f in (fill_nums):                       # ripeto analogamente a prima per ogni 
    print("")
    new_df = df[df.fill_num == f]
    # Definisco i nuovi vettori test e predictions
    test = new_df['test']
    predictions = new_df['predictions']
    lumi_inst = new_df['lumi_inst']
    test = test.to_numpy()
    predictions = predictions.to_numpy()
    lumi_inst = lumi_inst.to_numpy()
    print("len(lumi_inst)", len(lumi_inst))
    print("len(predict)", len(predictions))
    # Calcolo la somma delle luminosità istantanee
    somma_lumi = 0
    for m in range (len(lumi_inst)):
        somma_lumi = somma_lumi + lumi_inst[m]
    print("somma totale delle luminosità:  ", somma_lumi)
    # Riempio il vettore conteggi_test NON CONDSIDERANDO le correzioni alla trasparenza LSTM
    cont_misura_attuale = list()
    for i in range(0, num_bin):
        for j in range (0, len(test)):
            en_corr = energia[i] * test[j]
            if (en_corr > soglia):
                cont = cont + lumi_inst[j]
        #print(cont)
        cont_misura_attuale.append(cont) 
        cont = 0
    # Riempio il vettore conteggi_pred CONDSIDERANDO le correzioni alla trasparenza LSTM
    cont_corretto = list()
    for i in range(0, num_bin):
        for j in range (0, len(predictions)):
            en_corr = energia[i] * test[j] / predictions[j]
            if (en_corr > soglia):
                cont = cont + lumi_inst[j]
        #print(cont)
        cont_corretto.append(cont) 
        cont = 0
    # Creo un dataframe con l'energia e i conteggi con e senza correzioni
    df_result = pd.DataFrame(energia)
    df_result=df_result.assign(cont_corretto = 0)
    df_result.cont_corretto = pd.DataFrame(cont_corretto)
    df_result=df_result.assign(cont_misura_attuale = 0)
    df_result.cont_misura_attuale = pd.DataFrame(cont_misura_attuale)
    #print(df_result)
    # Scalo i conteggi per portarli tra 0 e 1
    df_result.cont_corretto = df_result.cont_corretto / somma_lumi
    df_result.cont_misura_attuale = df_result.cont_misura_attuale / somma_lumi
    # Rinomino le colonne del dataframe per renderlo più comprensibile
    df_result.set_axis(['energia', 'cont_corretto', 'cont_misura_attuale'], axis='columns', inplace=True)
    df_result=df_result.assign(fill_num = f)      # aggiungo una colonna contenente il numero del fill
    #print(df)
    # Plot delle turn on curve per singolo fill
    plt.plot(df_result.energia, df_result.cont_corretto, ".b-", label = "Predictions")
    plt.plot(df_result.energia, df_result.cont_misura_attuale, ".g-", label = "Test")
    plt.xlabel("Energy [GeV]")
    plt.ylabel(f"Fraction of events saved per instantaneous luminosity unit")
    plt.title(f"Trigger efficiency - Fill {f}")
    legend = plt.legend(['CM4S measure with predictions correction', 'CMS measure without predictions correction'], title = "Legend", loc="center left")
    plt.savefig(f"/home/marta/python/10-TurnOn/Turn_On_pesate_lumi_inst/PerFill/TurnOn_lumi_Fill_{f}.pdf")
    df_fills = df_fills.append(df_result, ignore_index=True)
    plt.show()

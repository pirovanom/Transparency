# Transparency

In questo repository sono contenuti i codici per fare predizioni sulla trasparenza dei cristalli di ECAL e le relative Turn On Curves. I codici sono da usare in sequenza: ognuno produce in output un file con estensione csv da usare come input per il codice successivo. 

fill_metadata_2017_10min.csv   +   iRing25new.npy     --->     Data_preparation.py     --->     input_data.csv   +   fill_n.csv   +   time.csv 

input_data.csv   +   fill_n.csv   +   time.csv     --->     LSTM.py     --->     output.csv

output.csv     --->     TurnOn_pesati_per_luminosita.py     --->     turn_on_lumi.csv

output.csv     --->     TurnOn_NON_pesate.py     --->     turn_on_originali.csv

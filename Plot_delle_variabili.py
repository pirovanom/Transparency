# DESCRIZIONE: questo codice fa il plot di tutte le variabili a disposizione. Mostra come evolvono nel tempo.

from pandas import read_csv
import matplotlib.pyplot as plt
import datetime
import numpy as np
import matplotlib.dates as mdates
import pandas as pd

# Importo i dati dal file csv
df = read_csv('/home/marta/python/5-Correlazioni/correlazioni.csv', header=0, index_col=0)
# Rinomino le colonne
df.set_axis(['time', 'lumi_inst', 'lumi_int', 'lumi_in_fill', 'lumi_since_last_point', 'lumi_last_fill', 'fill_num', 'time_in_fill', 'transparency'], axis='columns', inplace=True)
# Converto il tempo dal formato timestamp a data e ora
df.time = [datetime.datetime.fromtimestamp(ts) for ts in df.time]
# Cancello tutti i dati relativi a un fill diverso da quello selezionato
fill = 6323
df=df.drop(df[df.fill_num!=fill].index)

# Definisco le impostazioni per trasformare il tempo nel formato data e ora
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

values = df.values
# Divido le variabili da plottare in due gruppi (faccio due plot su due canvas, se no vengono troppo appiccicati)
groups_1 = [1, 2, 3, 4]
groups_2 = [5, 6, 7, 8]
colors = ["red", "orangered", "darkorange", "gold", "lawngreen", "springgreen", "aqua", "royalblue", "mediumpurple", "blue"]
colors=np.array(colors)
print(colors)

# Plot del primo gruppo
i = 1
plt.figure()
for group in groups_1:
	plt.subplot(len(groups_1), 1, i)
	#print(i, "  ", colors[i])
	plt.gca().xaxis.set_major_formatter(formatter)
	plt.gca().xaxis.set_major_locator(locator)
	plt.scatter(df.time, values[:, group],  color=colors[i])
	plt.gcf().autofmt_xdate()
	plt.title(df.columns[group], y=0.5, loc='right')
	i += 1
plt.suptitle('All parameters - fill 5872',fontsize=17)
plt.xlabel("Time")
plt.xticks(rotation='45')
plt.show()

# Plot del secondo gruppo
i = 1
plt.figure()
for group in groups_2:
	plt.subplot(len(groups_2), 1, i)
	plt.gca().xaxis.set_major_formatter(formatter)
	plt.gca().xaxis.set_major_locator(locator)
	plt.scatter(df.time, values[:, group],  color=colors[i+4])
	plt.gcf().autofmt_xdate()
	plt.title(df.columns[group], y=0.5, loc='right')
	i += 1
plt.suptitle('All parameters - fill 5872',fontsize=17)
plt.xlabel("Time")
plt.xticks(rotation='45')
plt.show()
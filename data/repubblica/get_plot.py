import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("sites_freq.csv") 
df.columns = ["sito","frequenza_sito"]

df = df[:10]
df = df.iloc[::-1]

ax = df.plot.barh(x='sito', y='frequenza_sito')
fig = ax.get_figure()
fig.savefig('diagramma.png', bbox_inches='tight')

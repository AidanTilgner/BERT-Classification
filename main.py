# !pip install kaggle pandas

from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

api = KaggleApi()
api.authenticate()

api.competition_download_file('sentiment-analysis-on-movie-reviews', 'train.tsv.zip', path="./")

df = pd.read_csv('./train.tsv', sep='\t')

df.drop_duplicates(subset="SentenceId", keep="first", inplace=True)
print(df.head())
print(len(df))

seqlen = df["Phrase"].apply(lambda x: len(x.split()))

sns.set_style("darkgrid")
plt.figure(figsize=(16, 10))
sns.displot(seqlen)
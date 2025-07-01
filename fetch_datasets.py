#%%
from sklearn.datasets import fetch_rcv1
dataset = fetch_rcv1()

print(dataset.data.shape)

#%%
from sklearn.datasets import fetch_20newsgroups
cats = ['alt.atheism', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
list(newsgroups_train.target_names)
['alt.atheism', 'sci.space']
newsgroups_train.filenames.shape
(1073,)
newsgroups_train.target.shape
(1073,)
newsgroups_train.target[:10]
#array([0, 1, 1, 1, 0, 1, 1, 0, 0, 0])
# %%

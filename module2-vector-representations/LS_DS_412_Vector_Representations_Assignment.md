
<img align="left" src="https://lever-client-logos.s3.amazonaws.com/864372b1-534c-480e-acd5-9711f850815c-1524247202159.png" width=200>
<br></br>

# Vector Representations
## *Data Science Unit 4 Sprint 2 Assignment 2*


```python
import re
import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import spacy

from bs4 import BeautifulSoup
```

## 1) *Clean:* Job Listings from indeed.com that contain the title "Data Scientist" 

You have `job_listings.csv` in the data folder for this module. The text data in the description column is still messy - full of html tags. Use the [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) library to clean up this column. You will need to read through the documentation to accomplish this task. 


```python
from bs4 import BeautifulSoup
import requests

##### Your Code Here #####
df = pd.read_csv('job_listings.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>description</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>b"&lt;div&gt;&lt;div&gt;Job Requirements:&lt;/div&gt;&lt;ul&gt;&lt;li&gt;&lt;p&gt;...</td>
      <td>Data scientist</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>b'&lt;div&gt;Job Description&lt;br/&gt;\n&lt;br/&gt;\n&lt;p&gt;As a Da...</td>
      <td>Data Scientist I</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>b'&lt;div&gt;&lt;p&gt;As a Data Scientist you will be work...</td>
      <td>Data Scientist - Entry Level</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>b'&lt;div class="jobsearch-JobMetadataHeader icl-...</td>
      <td>Data Scientist</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>b'&lt;ul&gt;&lt;li&gt;Location: USA \xe2\x80\x93 multiple ...</td>
      <td>Data Scientist</td>
    </tr>
  </tbody>
</table>
</div>




```python
#df = df.drop(columns = 'Unnamed: 0')
df['text'] = [BeautifulSoup(text).get_text() for text in df['description'] ]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>description</th>
      <th>title</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b"&lt;div&gt;&lt;div&gt;Job Requirements:&lt;/div&gt;&lt;ul&gt;&lt;li&gt;&lt;p&gt;...</td>
      <td>Data scientist</td>
      <td>b"Job Requirements:\nConceptual understanding ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b'&lt;div&gt;Job Description&lt;br/&gt;\n&lt;br/&gt;\n&lt;p&gt;As a Da...</td>
      <td>Data Scientist I</td>
      <td>b'Job Description\n\nAs a Data Scientist 1, yo...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b'&lt;div&gt;&lt;p&gt;As a Data Scientist you will be work...</td>
      <td>Data Scientist - Entry Level</td>
      <td>b'As a Data Scientist you will be working on c...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b'&lt;div class="jobsearch-JobMetadataHeader icl-...</td>
      <td>Data Scientist</td>
      <td>b'$4,969 - $6,756 a monthContractUnder the gen...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b'&lt;ul&gt;&lt;li&gt;Location: USA \xe2\x80\x93 multiple ...</td>
      <td>Data Scientist</td>
      <td>b'Location: USA \xe2\x80\x93 multiple location...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.drop(columns='description')
```

## 2) Use Spacy to tokenize the listings 


```python
##### Your Code Here #####
import spacy
from spacy.tokenizer import Tokenizer
nlp = spacy.load("en_core_web_sm")

# Tokenizer
tokenizer = Tokenizer(nlp.vocab)
```


```python
# Tokenizer Pipe

tokens = []

""" Make them tokens """
for doc in tokenizer.pipe(df['text'], batch_size=500):
    doc_tokens = [token.text for token in doc]
    tokens.append(doc_tokens)

df['tokens'] = tokens
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>text</th>
      <th>tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Data scientist</td>
      <td>b"Job Requirements:\nConceptual understanding ...</td>
      <td>[b"Job, Requirements:\nConceptual, understandi...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Data Scientist I</td>
      <td>b'Job Description\n\nAs a Data Scientist 1, yo...</td>
      <td>[b'Job, Description\n\nAs, a, Data, Scientist,...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Data Scientist - Entry Level</td>
      <td>b'As a Data Scientist you will be working on c...</td>
      <td>[b'As, a, Data, Scientist, you, will, be, work...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Data Scientist</td>
      <td>b'$4,969 - $6,756 a monthContractUnder the gen...</td>
      <td>[b'$4,969, -, $6,756, a, monthContractUnder, t...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Data Scientist</td>
      <td>b'Location: USA \xe2\x80\x93 multiple location...</td>
      <td>[b'Location:, USA, \xe2\x80\x93, multiple, loc...</td>
    </tr>
  </tbody>
</table>
</div>



## 3) Use Scikit-Learn's CountVectorizer to get word counts for each listing.


```python
##### Your Code Here #####
from sklearn.feature_extraction.text import CountVectorizer

# list of text documents
text = list(df.text)

# create the transformer
vect = CountVectorizer()

# build vocab
vect.fit(text)

# transform text
dtm = vect.transform(text)

# Create a Vocabulary
# The vocabulary establishes all of the possible words that we might use.

# The vocabulary dictionary does not represent the counts of words!!
print(vect.get_feature_names()[-10:])
```

    ['zenreach', 'zero', 'zeus', 'zf', 'zheng', 'zillow', 'zones', 'zoom', 'zuckerberg', 'zurich']



```python
# Get Word Counts for each document
dtm = pd.DataFrame(dtm.todense(), columns=vect.get_feature_names())
dtm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>00</th>
      <th>000</th>
      <th>02115</th>
      <th>03</th>
      <th>0356</th>
      <th>04</th>
      <th>062</th>
      <th>06366</th>
      <th>08</th>
      <th>10</th>
      <th>...</th>
      <th>zenreach</th>
      <th>zero</th>
      <th>zeus</th>
      <th>zf</th>
      <th>zheng</th>
      <th>zillow</th>
      <th>zones</th>
      <th>zoom</th>
      <th>zuckerberg</th>
      <th>zurich</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>421</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>422</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>423</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>424</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>425</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>426 rows × 10069 columns</p>
</div>



## 4) Visualize the most common word counts


```python
##### Your Code Here #####
from collections import Counter

def count(docs):

        word_counts = Counter()
        appears_in = Counter()
        
        total_docs = len(docs)

        for doc in docs:
            word_counts.update(doc)
            appears_in.update(set(doc))

        temp = zip(word_counts.keys(), word_counts.values())
        
        wc = pd.DataFrame(temp, columns = ['word', 'count'])

        wc['rank'] = wc['count'].rank(method='first', ascending=False)
        total = wc['count'].sum()

        wc['pct_total'] = wc['count'].apply(lambda x: x / total)
        
        wc = wc.sort_values(by='rank')
        wc['cul_pct_total'] = wc['pct_total'].cumsum()

        t2 = zip(appears_in.keys(), appears_in.values())
        ac = pd.DataFrame(t2, columns=['word', 'appears_in'])
        wc = ac.merge(wc, on='word')

        wc['appears_in_pct'] = wc['appears_in'].apply(lambda x: x / total_docs)
        
        return wc.sort_values(by='rank')
    
wc = count(df.tokens)
wc.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>appears_in</th>
      <th>count</th>
      <th>rank</th>
      <th>pct_total</th>
      <th>cul_pct_total</th>
      <th>appears_in_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>59</th>
      <td>and</td>
      <td>425</td>
      <td>11477</td>
      <td>1.0</td>
      <td>0.058473</td>
      <td>0.058473</td>
      <td>0.997653</td>
    </tr>
    <tr>
      <th>22</th>
      <td>to</td>
      <td>422</td>
      <td>6736</td>
      <td>2.0</td>
      <td>0.034319</td>
      <td>0.092792</td>
      <td>0.990610</td>
    </tr>
    <tr>
      <th>307</th>
      <td>the</td>
      <td>414</td>
      <td>4931</td>
      <td>3.0</td>
      <td>0.025123</td>
      <td>0.117915</td>
      <td>0.971831</td>
    </tr>
    <tr>
      <th>1</th>
      <td>of</td>
      <td>420</td>
      <td>4532</td>
      <td>4.0</td>
      <td>0.023090</td>
      <td>0.141005</td>
      <td>0.985915</td>
    </tr>
    <tr>
      <th>78</th>
      <td>in</td>
      <td>421</td>
      <td>3436</td>
      <td>5.0</td>
      <td>0.017506</td>
      <td>0.158511</td>
      <td>0.988263</td>
    </tr>
  </tbody>
</table>
</div>




```python
import squarify
import matplotlib.pyplot as plt

wc_top20 = wc[wc['rank'] <= 20]

squarify.plot(sizes=wc_top20['pct_total'], label=wc_top20['word'], alpha=.8 )
plt.axis('off')
plt.show()
```


![png](output_14_0.png)


## 5) Use Scikit-Learn's tfidfVectorizer to get a TF-IDF feature matrix


```python
##### Your Code Here #####
from sklearn.feature_extraction.text import TfidfVectorizer

# Instantiate vectorizer object
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

# Create a vocabulary and get word counts per document
# Similiar to fit_predict
dtm = tfidf.fit_transform(df.text)

# Print word counts

# Get feature names to use as dataframe column headers
dtm = pd.DataFrame(dtm.todense(), columns=tfidf.get_feature_names())

# View Feature Matrix as DataFrame
dtm.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>000</th>
      <th>04</th>
      <th>10</th>
      <th>100</th>
      <th>1079302</th>
      <th>11</th>
      <th>12</th>
      <th>125</th>
      <th>14</th>
      <th>15</th>
      <th>...</th>
      <th>years</th>
      <th>yearthe</th>
      <th>yes</th>
      <th>yeti</th>
      <th>york</th>
      <th>young</th>
      <th>yrs</th>
      <th>zeus</th>
      <th>zf</th>
      <th>zillow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.093431</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 5000 columns</p>
</div>



## 6) Create a NearestNeighbor Model. Write the description of your ideal datascience job and query your job listings. 


```python
##### Your Code Here #####
from sklearn.neighbors import NearestNeighbors

# Fit on DTM
nn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree')    #also try ball_tree
nn.fit(dtm)
```




    NearestNeighbors(algorithm='kd_tree', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     radius=1.0)




```python
nn.kneighbors([dtm.iloc[122]])
```




    (array([[0.        , 0.        , 1.22207044, 1.22207044, 1.22889549]]),
     array([[156,  41, 184, 147, 122]]))




```python
df.text[122][:200]
```




    "b'Senior Data Scientist-19000BN4\\n\\n\\nPreferred Qualifications\\n\\n(SENIOR) DATA SCIENTIST, CLIENT ANALYTICS\\n\\nOracle Data Cloud helps marketers use data to reach the right audiences, capture consumer"




```python
job_desc = [ """
I'm looking for a Data Scientist role at a company with strong leadership and an ambitious mission.

The leaders should offer mentorship when appropriate but also allow me independence in day to day responsibilities.

The ideal company would be highly innovative and a leader within its field. 

I would like my primary role to be producing strategy driven analysis of a market.

This could be for marketing purposes, businness intelligence, or for resource allocation purposes.
"""]
```


```python
new = tfidf.transform(job_desc)
```


```python
new
```




    <1x5000 sparse matrix of type '<class 'numpy.float64'>'
    	with 34 stored elements in Compressed Sparse Row format>




```python
# index numbers relate to the tfidf above
# is the higher numbers or lower numbers better?
nn.kneighbors(new.todense())
```




    (array([[1.30995857, 1.31933775, 1.33838557, 1.3450627 , 1.34635325]]),
     array([[358, 405, 319, 284,  59]]))




```python
df.text[358]
```




    "b'What you\\xe2\\x80\\x99ll be doing\\xe2\\x80\\xa6\\nThe primary focus of the Data Scientist in the Data Science and Engineering group is to influence the long-term direction of our product line using data science and to drive innovation in bringing data science into our product line. To do this, the Data Scientist will partner with teams to solve internal data and analysis problems, building best practices and innovative tools to help scale out the use of data science among teams.\\nWhat some of your responsibilities include\\xe2\\x80\\xa6\\nPartner with teams on modeling and analysis problems \\xe2\\x80\\x93 from transforming problem statements into analysis problems, to working through data modeling and engineering, to analysis and communication of results.\\nBuild up and share best practices for teams working on analytical problems\\nUse experience gained in the above and expertise in this space to influence our product roadmap, potentially working with prototype engineering team to add additional capabilities to our products to solve more of these problems.\\nWho you are\\xe2\\x80\\xa6\\nEducated. PhD in Statistics or a Statistics-related field ( e.g. econometrics, operations research, machine learning, etc.). Excellent candidates with a Masters degree or education through Data-science bootcamps will also be considered.\\nExperienced. Alternately, 3+ years of work in this space, with deep knowledge of one or more data science platforms, including R or Python/pandas/ scikit -learn. Additional knowledge of other commercial platforms desired.\\nTechnically Savvy. Moderate coding ability desired in languages suitable for engineering purposes .\\nKnowledgeable. Familiarity with both purely predictive and interpretable models, including Generalized Linear Models. Ability to walk through modeling problems \\xe2\\x80\\x93 transforming a problem into a data analysis problem, understanding the caveats, and clearly communicating what the results mean.\\nA True Team Player. Demonstrated self-starter ability, and experience with ambiguity and fast-moving teams.\\nYou are a Recruiter! Tableau hires company builders and, in this role, you will be asked to be on the constant lookout for the best talent to bring onboard to help us continue to build one of the best companies in the world!\\n#LI-EF1\\nTableau Software is an Equal Opportunity Employer.\\nTableau Software is a company on a mission. We help people see and understand their data. After a highly successful IPO in 2013, Tableau has become a market-defining company in the business intelligence industry. Our culture is casual and high-energy. We are passionate about our product and our mission and we are loyal to each other and our company. We value work/life balance, efficiency, simplicity, freakishly friendly customer service, and making a difference in the world!'"




```python
df.text[284]
```




    "b'Socure is headquartered in NYC and is the leader in machine learning driven digital identity verification. The company\\xe2\\x80\\x99s predictive analytics platform applies artificial intelligence and machine learning to trusted online/offline sources to verify identities in real-time across the web with the singular mission to become the trusted source of identities on the internet.\\n\\nSocure is looking for an experienced Pre-Sales Data Scientist to help drive Identity Verification solutions and create go-forward plans with our product team. The ideal candidate must be able to act as a consultant to identify customer needs, understand the best path to solve these, explain/gather data requirements, deliver running data tests and work closely with the Sales personnel to present results and final recommendation to clients. A strong background in data analytics, data science, and/or fraud analysis is required.\\n\\nResponsibilities:\\n\\nWork closely with sales leadership to determine clients pain points\\nStructure proof-of-concept tests and act as the lead data expert ensuring that the\\nRight data is collected, analyzed and presented efficiently.\\nCreate analytical customer-facing results material from analyzing proof-of-concept results.\\nHelp deliver the best possible solution based on results and subsequent customer feedback.\\nGrow department and manage additional resources, while maintaining a player-coach posture.\\nQualifications:\\n\\nBachelor Degree in CS, Mathematics, Statistics or similar or equivalent work experience\\nExperience with SQL, R and/or Python, Excel, and PowerPoint\\nA minimum of 3+ years of industry experience working in a similar role.\\nKnowledge of UNIX systems management, Java, Scala, Python, C/C++, SQL, HQL, SPARQL for RDF\\nUnderstanding of Spark, Hadoop/Hive, StarDog, Amazon Redshift, MongoDB.\\nStrong understanding of data mining, machine learning, statistical techniques, and underlying algorithms to discuss with clients\\nAbility to multi-task efficiently and think on the spot when being in front of clients.\\nA few of the many perks we offer\\n\\nCompetitive base salary\\nEquity - every employee is a stakeholder in our enormous upside\\nA tech-first company culture driven by entrepreneurial thinking and talent\\nA group of highly intelligent peers that are all working in unison towards the same mission\\nTransparency is what our product is built on\\xe2\\x80\\x94and so is our culture\\nGenerous medical, dental and vision benefits for employees and their dependents\\nFlexible PTO\\n401K with company match\\n...and so much more\\nWe are an equal opportunity employer and value diversity of all kinds at our company.\\nWe do not discriminate on the basis of race, religion, color, national origin, gender,\\nsexual orientation, age, marital status, veteran status, or disability status.'"



## Stretch Goals

 - Try different visualizations for words and frequencies - what story do you want to tell with the data?
 - Scrape Job Listings for the job title "Data Analyst". How do these differ from Data Scientist Job Listings
 - Try and identify requirements for experience specific technologies that are asked for in the job listings. How are those distributed among the job listings?
 - Use a clustering algorithm to cluster documents by their most important terms. Do the clusters reveal any common themes?
  - **Hint:** K-means might not be the best algorithm for this. Do a little bit of research to see what might be good for this. Also, remember that algorithms that depend on Euclidean distance break down with high dimensional data.
 - Create a labeled dataset - which jobs will you apply for? Train a model to select the jobs you are most likely to apply for. :) 

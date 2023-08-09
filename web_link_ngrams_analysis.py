import pylab as p
import requests as rq
from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
import lxml.html
import string
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist, bigrams
from nltk import ngrams

# Using Xpath to exctract link from wikipedia page

url = 'https://en.wikipedia.org/wiki/Python_(programming_language)'
wiki_text = urlopen(url).read().decode('utf-8')

tree= lxml.html.fromstring(wiki_text)

wiki_url = 'https://en.wikipedia.org'

link_elements = tree.xpath('//*[@id="mw-content-text"]/div[1]/div[7]//a')
links = [wiki_url + link.attrib['href'] for link in link_elements]

# Collecting plain text from defined links

all_text = []
for link in links:
    html = rq.get(link).text
    soup = bs(html)
    paragraph_text = soup.find_all('p')
    all_text.extend(p.text for p in paragraph_text)

text = ' '.join(all_text)


# Finding Ngrams

en_stopwords = set(nltk.corpus.stopwords.words('english'))
translator = str.maketrans('', '', string.punctuation + string.digits)
cleaned_text = text.translate(translator)
words = cleaned_text.lower().split()
cleaned_words = [w for w in words if w not in en_stopwords and len(w) > 3]

ng = [' '.join(ngr) for ngr in ngrams(cleaned_words, 2)]
ng_fd = FreqDist(ng)

ng_fd.plot(20,)
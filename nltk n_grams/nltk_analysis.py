from pdfminer.high_level import extract_text
from glob import glob
import os
import nltk
from nltk.util import ngrams
import string
import matplotlib.pyplot as plt

en_stopwords = set(nltk.corpus.stopwords.words('english'))


def create_pdf_ngrams(path, extension="pdf"):
    """Takes a path to a folder with .pdf files, reads and cleans text,
    then plots unigram, bigram and trigram frequency distributions"""

    docs = glob(os.path.join(path, f"*.{extension}"))
    text =' '.join(extract_text(w) for w in docs)
    lines = text.split("\n")
    cleaned_lines = []
    for l in lines:
        if len(l)==0:
            continue
        elif l[-1]=="-":
            cleaned_lines.append(l[:-1])
        else:cleaned_lines.append(l + ' ')
    text = ''.join(cleaned_lines)

    # remove punctuation, numbers, stopwords
    translator = str.maketrans('', '', string.punctuation + string.digits)
    text = text.translate(translator)
    words = text.lower().split()
    words = [w for w in words if w not in en_stopwords and len(w) > 3]

    unigram_fd = nltk.FreqDist(words)

    bigrams = list([' '.join(bg) for bg in nltk.bigrams(words)])
    bigram_fd = nltk.FreqDist(bigrams)

    n_gram = list([' '.join(bg) for bg in nltk.ngrams(words,3)])
    n_gram_fd = nltk.FreqDist(n_gram)


    unigram_fd.plot(20)

    bigram_fd.plot(20)

    n_gram_fd.plot(20)

create_pdf_ngrams('C:/Users/Jaroslav/anaconda3/exercises/nltk n_grams/', extension='pdf')


""""
Understanding N-grams:
Text n-grams are commonly utilized in natural language processing and 
text mining. It’s essentially a string of words that appear in the same 
window at the same time.

Using these n-grams and the probabilities of the occurrences of certain 
words in certain sequences could improve the predictions of auto completion 
systems.
"""






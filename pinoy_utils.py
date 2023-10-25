import re
import string
import numpy as np
from nltk.tokenize import TweetTokenizer
from TagalogStemmer.TglStemmer import stemmer as tgl_stemmer

# Replace with your actual list of Filipino stopwords
filipino_stopwords = ['akin', 'aking', 'ako', 'alin', 'amin', 'ang', 'ano', 'anuman', 'at', 'ating', 'bago', 'bakit', 'bilang', 'bawat', 
                      'dahil', 'dalawa', 'dapat', 'din', 'dito', 'doon', 'gagawin', 'gayunman', 'ginagawa', 'ginawa', 'gusto', 'naman'
                      ,'ka', 'kaya', 'kaysa', 'kong', 'kulang', 'kumuha', 'laban', 'lahat', 'muli', 'nais', 'nasaan', 'narito']

def process_article(article):
    """Process article function.
    Input:
        article: a string containing an article
    Output:
        article_clean: a list of words containing the processed article
    """
    
    # Convert to lowercase
    article = article.lower()
    
    # Remove punctuation and special characters
    article = re.sub(f'[{re.escape(string.punctuation)}]', '', article)
    
    # Remove single characters
    article = re.sub(r'\b\w\b', '', article)
    
    # Substitute multiple spaces with a single space
    article = re.sub(r'\s+', ' ', article)
    
    # Tokenize articles
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    article_token = tokenizer.tokenize(article)
    
    article_clean = []
    for salita in article_token:
        # Use the Filipino stemmer to stem the word
        stemmed_salita = tgl_stemmer('2', salita, '1')
        # Check if the stemmed word is not a stopword
        if stemmed_salita not in filipino_stopwords:
            article_clean.append(stemmed_salita)
    
    return article_clean

def build_freqs(articles, ys):
    """Build frequencies.
    Input:
        articles: a list of articles
        ys: an m x 1 array with the real/fake label of each article
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (salita, real/fake) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all articles
    # and over all processed words in each article.
    freqs = {}
    for y, article in zip(yslist, articles):
        for salita in process_article(article):
            #Convert salita to string 2 point

            salita2 = ""
            for i in salita:
                salita2 += i

            #Use Regex to delete "[]" 4 points

            salita2 = re.sub(r'[|]', '', salita2)

            pair = (salita2, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs
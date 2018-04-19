from nltk import word_tokenize
from gensim import corpora
from gensim import models
from gensim.models import LdaModel, LdaMulticore
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


car_df = pd.read_pickle('ads_image.pkl')
docs = np.array(car_df['ocr_clean'])
docs = [document.split() for document in docs]

dictionary = Dictionary(docs)
dictionary.filter_extremes(no_below=10, no_above=.1)
corpus = [dictionary.doc2bow(doc) for doc in docs]
print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))


def evaluate_graph(dictionary, corpus, texts, begin, end, steps):
    """
    Function to display num_topics - LDA graph using c_v coherence

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit
    
    Returns:
    -------
    lm_list : List of LDA topic models
    """
    u_mass = []
    c_v = []
    lm_list = []
    for num_topics in range(begin, end, steps):
        lm = LdaMulticore(corpus=corpus, num_topics=num_topics, workers=24, id2word=dictionary, eval_every=10, eta='auto', passes=20)
        lm_list.append(lm)
        cm_umass = CoherenceModel(model=lm, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        cm_cv = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm_cv.get_coherence())
        u_mass.append(cm_umass.get_coherence())
    print(c_v)
    file_1 = open('c_v.txt', 'w')
    for item in c_v:
    	file_1.write("%s\n" % item)

    print(u_mass)
    file_2 = open('u_mass.txt', 'w')
    for item in u_mass:
    	file_2.write("%s\n" % item)
        
    # Show graph
    #x = range(begin, end, steps)
    #plt.plot(x, c_v)
    #plt.xlabel("num_topics")
    #plt.ylabel("Coherence score")
    #plt.legend(("c_v"), loc='best')
    #plt.savefig('c_v_topics.png', dpi=300)

    # Show graph
    #x = range(begin, end, steps)
    #plt.plot(x, u_mass)
    #plt.xlabel("num_topics")
    #plt.ylabel("Coherence score")
    #plt.legend(("u_mass"), loc='best')
    #plt.savefig('u_mass_topics.png', dpi=300)
    
    #return lm_list

c_v = evaluate_graph(dictionary, corpus, docs, 5, 100, 1)


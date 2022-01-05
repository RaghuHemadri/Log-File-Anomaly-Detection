from gensim import utils
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import correlation as cosine
import logging
def clean_text(s, filters):
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s

def barplot(words, words_counts, title):
    fig = plt.figure(figsize=(18,6))
    bar_plot = sns.barplot(x=words, y=words_counts)
    for item in bar_plot.get_xticklabels():
        item.set_rotation(90)
    plt.title(title)
    plt.show()

def key_word_counter(tupple):
    return tupple[1]

def key_consine_similarity(tupple):
    return tupple[1]

def get_computed_similarities(vectors, predicted_vectors, df, reverse=False):
    data_size = len(df)
    cosine_similarities = []
    for i in range(data_size):
        cosine_sim_val = (1 - cosine(vectors[i], predicted_vectors[i]))
        cosine_similarities.append((i, cosine_sim_val))

    return sorted(cosine_similarities, key=key_consine_similarity, reverse=reverse)

def display_top_n(sorted_cosine_similarities, df, n=5):
    logging.basicConfig(filename="result/results.log",
                    format='%(message)s',
                    filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    for i in range(n):
        index, consine_sim_val = sorted_cosine_similarities[i]
        msg = 'Level: ' + str(df.iloc[index, 0]) + '; Content: ' + str(df.iloc[index, 1])
        logger.info(msg)
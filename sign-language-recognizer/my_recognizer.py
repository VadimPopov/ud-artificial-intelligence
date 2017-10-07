import warnings
from asl_data import SinglesData

def recognize(models: dict, test_set: SinglesData):
    """Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key is a word and value is Log Likelihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... }]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    Xlengths = test_set.get_all_Xlengths()

    for idx in Xlengths:
        probabilities.append({})
        X, lengths = Xlengths[idx]
        for word in models:
            try:
                probabilities[idx][word] = models[word].score(X, lengths)
            except:
                pass

    for prob in probabilities:
        highest_ll = max(prob.values())
        for word in prob:
            if prob[word] == highest_ll:
                guesses.append(word)
                break

    return probabilities, guesses

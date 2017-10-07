import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

class ModelSelector(object):
    '''base class for model selection (strategy design pattern)'''
    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3, min_n_components=2, max_n_components=10, random_state=14,
                 verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

class SelectorConstant(ModelSelector):
    """select the model with value self.n_constant"""
    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)

class SelectorBIC(ModelSelector):
    """select the model with the lowest Bayesian Information Criterion(BIC) score:
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def select(self):
        """select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        models = []
        for n_comp in range(self.min_n_components, self.max_n_components + 1):
            n_params = n_comp ** 2 + 2 * n_comp * len(self.X[0]) - 1
            try:
                model = self.base_model(n_comp)
                log_l = model.score(self.X, self.lengths)
                bic = -2 * log_l + n_params * np.log(len(self.X))
                models.append((bic, model))
            except:
                pass

        if models:
            _, best_model = min(models)
            return best_model
        else:
            return self.base_model(self.n_constant)

class SelectorDIC(ModelSelector):
    """select best model based on Discriminative Information Criterion
    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    """
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        models = []
        other_log_l = []

        for n_comp in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_comp)
                log_l = model.score(self.X, self.lengths)
                for word in self.words:
                    if word == self.this_word: continue
                    X, lengths = self.hwords[word]
                    other_ll = model.score(X, lengths)
                    other_log_l.append(other_ll)
                dic = log_l - np.mean(other_log_l)
                models.append((dic, model))
            except:
                pass

        if models:
            _, best_model = max(models)
            return best_model
        else:
            return self.base_model(self.n_constant)

class SelectorCV(ModelSelector):
    """select best model based on average log Likelihood of cross-validation folds"""
    def mean_cv_likelihood(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        cv_log_l = []
        splitter = KFold()
        self.X = np.array(self.X)

        for train_idx, test_idx in splitter.split(self.X):
            if len(train_idx) >= num_states:
                try:
                    hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(self.X[train_idx])
                    log_l = hmm_model.score(self.X[test_idx])
                    cv_log_l.append(log_l)
                except:
                    pass

        return np.mean(cv_log_l)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        models = []

        for n_comp in range(self.min_n_components, self.max_n_components + 1):
            mean_cv_log_l = self.mean_cv_likelihood(n_comp)
            models.append((mean_cv_log_l, n_comp))

        if models:
            _, best_n_comp = max(models)
            best_model = self.base_model(best_n_comp)
            return best_model
        else:
            return self.base_model(self.n_constant)

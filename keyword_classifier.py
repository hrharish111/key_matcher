import os
import re
import string

import nltk
import numpy as np
import wikiwords
from gensim import corpora, models
from nltk import ngrams
from nltk.tokenize import sent_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV
from textblob import TextBlob

from keyword_machine import node_degree_extraction

TFIDF_MDL_PATH = os.getcwd() + '/keyword_machine/modeldir/tfidf.mm'
DICT_PATH = os.getcwd() + '/keyword_machine/modeldir/dict.mm'
CLF_DICT_PATH = os.getcwd() + '/keyword_machine/modeldir/clf_dict.mm'
CLF_MDL_PATH = os.getcwd() + '/keyword_machine/modeldir/clf_mdl.mm'
NER_TAGGER_JAR_LOC = os.getcwd() + '/keyword_machine/modeldir/english.conll.4class.caseless.distsim.crf.ser.gz'
NER_TAGGER_MODEL_LOC = os.getcwd() + '/keyword_machine/modeldir/stanford-ner.jar'
TRAIN_VEC_FILE_LOC = os.getcwd() + '/keyword_machine/modeldir/train_vec.mm'

CAPS_REGEX = re.compile('[A-Z]\w+')


def scikit_load_model(filename):
    return joblib.load(filename)


def scikit_save_model(mdl, filename):
    joblib.dump(mdl, filename, compress=3)


# ..........................features..............................#
orthographic_feature = lambda freq, tfidf_score, length, first_loc, occur_std_dev, num_caps: {
    'local_word_freq': freq, 'tfidf_score': tfidf_score, 'len': length, 'first_loc': first_loc,
    'occurence_std_dev': occur_std_dev, 'is_caps': num_caps}

lexical_feature = lambda noun_phrase, named_entity: {'noun_phrase': noun_phrase, 'named_entity': named_entity}

wiki_features = lambda tfidf_score: {'wiki_df_score': tfidf_score}


def remove_punctuation(text):
    """
    Returns text free of punctuation marks
    """
    exclude = set(string.punctuation)
    return ''.join([ch for ch in text if ch not in exclude])


class KeyWordClassifier(object):
    """description of class"""

    def __init__(self, tfidf_model_loc, dict_loc, ner_tagger_model_loc, ner_tagger_jar_loc, clf_dict_path,
                 clf_model_path):
        grammar = r"""
          NBAR:
              {<NN.*|JJ>*<NN.*>}

          NP:
              {<NBAR>}
              {<NBAR><IN><NBAR>}   # from Alex Bowe's nltk tutorial
        """
        self.tfidf_model = scikit_load_model(tfidf_model_loc)
        self.dict = scikit_load_model(dict_loc)
        self.dict_v = None
        self.dict_v_path = clf_dict_path
        self.model = None
        self.model_path = clf_model_path
        # self.st = StanfordNERTagger(ner_tagger_model_loc,ner_tagger_jar_loc)
        # self.chunker = nltk.RegexpParser(grammar)
        # self.parser = English()
        self.max_wiki_freq = wikiwords.freq('the')
        self.max_word_freq = max(self.dict.dfs.values())
        self.vec_file_loc = TRAIN_VEC_FILE_LOC

    def get_trigrams(text, num_trigrams):
        """
        Return all members of most frequent trigrams
        """
        trigram_measures = nltk.collocations.TrigramAssocMeasures()
        finder = nltk.collocations.TrigramCollocationFinder.from_words(text.lower().split())
        finder.apply_freq_filter(1)  # ignore trigrams that occur only once
        top_ngrams = finder.nbest(trigram_measures.pmi, num_trigrams)

        ngrams = []
        for ng in top_ngrams:
            ngrams.extend(list(ng))

        ngrams = [remove_punctuation(n) for n in list(set(ngrams))]
        ngrams = [n for n in ngrams if n]
        return ngrams

    def _get_wiki_bigrams_trigrams(self, text, min_freq=10):
        """
        bigrams or trigrams which are present in wiki
        """
        sentences = sent_tokenize(text)
        valid_ngrams = []
        for sentence in sentences:
            for gram in [2, 3]:
                tokens = ngrams(sentence.split(), gram)
                sel_ngrams = [wikiwords.freq(token) > min_freq for token in tokens]
                valid_ngrams.extend(sel_ngrams)

        return valid_ngrams

    def get_nounphrases(self, text):
        """
        Returns noun phrases in text
        """
        grammar = r"""
        NBAR:
            {<NN.*|JJ>*<NN.*>}

        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}   # from Alex Bowe's nltk tutorial
      """

        sentences = nltk.sent_tokenize(text.lower())
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        sentences = [nltk.pos_tag(sent) for sent in sentences]

        noun_phrases = []
        for sent in sentences:
            tree = chunker.parse(sent)
            for subtree in tree.subtrees():
                if subtree.label() == 'NP':
                    noun_phrases.extend([w[0] for w in subtree.leaves()])

        noun_phrases = [remove_punctuation(nphrase) for nphrase in noun_phrases]
        noun_phrases = [n for n in noun_phrases if n]
        return noun_phrases

    def get_named_entities(self, text):

        ner_tagged = self.st.tag(text.lower().split())
        named_entities = []
        if len(ner_tagged) > 0:
            for n in ner_tagged:
                if n[1] != 'O':
                    named_entities.append(remove_punctuation(n[0]))
        named_entities = [n for n in named_entities if n]
        return named_entities

    def get_lexical_features(self, candidates, text, named_entities, noun_phrases):
        info_list = []
        for candidate in candidates:
            named_entity = True if candidate in named_entities else False
            noun_phrase = True if candidate in noun_phrases else False
            lex_feat = lexical_feature(noun_phrase, named_entity)
            info_list.append(lex_feat)

    def get_wiki_features(self, candidates):
        """
        """
        # only wiki tfidf scores are added now rest must be added later

        wiki_features_dict = {w.lower(): wiki_features(wikiwords.freq(w.lower()) / float(self.max_wiki_freq)) for w in
                              candidates}
        # key_phraseness=

        # node degree
        # node_features = node_degree_extraction.node_degree_caluclation(candidates)
        # for key,value in node_features.items():
        #     wiki_features_dict[key.lower()].update(value)
        #
        # print(wiki_features_dict)
        return wiki_features_dict

    def get_orthographic_features(self, candidates, text):
        """
        these feature are from current text
        tfidf score,freq,mean and std dev between occurence,first occurence location
        """
        info_dict = {}
        num_docs = self.tfidf_model.num_docs
        text = text.lower()
        text_size = len(text)
        num_tokens = len(re.split('\W+', text))
        for candidate in candidates:
            # change here for dict
            try:
                num_caps = len(CAPS_REGEX.findall(candidate))
                candidate = candidate.lower()
                id = self.dict.token2id[candidate]
                norm_freq = self.tfidf_model.dfs[id] / num_docs
                # normalized idf score
                norm_idf = self.tfidf_model.idfs[id] / num_docs
                norm_tfidf = norm_freq * norm_idf
                locations = re.finditer(candidate, text)
                locations = [loc.start() for loc in locations]
                if len(locations) > 1:
                    inc_diff = np.ediff1d(locations)
                    mean_diff = np.mean(inc_diff) / text_size
                else:
                    mean_diff = 0
                first_loc = locations[0] / text_size
                local_df = len(locations) / num_tokens
                orth_feature = orthographic_feature(local_df, norm_tfidf, len(candidate), first_loc, mean_diff,
                                                    num_caps)
                info_dict[candidate] = orth_feature
            except Exception as e:
                print(e)
        return info_dict

    def generate_candidates(self, text):
        bi_tri_grams = self._get_wiki_bigrams_trigrams(text)
        named_entities = self.get_named_entities(text)
        noun_phrases = self.get_nounphrases(text)
        return list(set.union(set(bi_tri_grams), set(named_entities), set(noun_phrases)))

    def get_features(self, candidates, text):
        # is candidate wiki bigram/trigram

        pass

    def get_noun_pharase_named_entities(self, doc):

        pass

    def extract_noun_adjective(self, doc, pos_filter=['NN', 'JJ']):
        tokens = TextBlob(doc)
        noun_adj_list = []
        for token_tag in tokens.tags:
            for ref_tag in pos_filter:
                if token_tag[1].startswith(ref_tag):
                    noun_adj_list.append(token_tag[0])
        return noun_adj_list

    def filter_tokens(self, candidates, noun_adj_list):
        """
        add if any other type of filter u want to use.see if lemmatization etc required.
        """
        return [candidate for candidate in candidates if candidate in noun_adj_list and len(candidate) >= 3]

    def extract_features_from_text(self, doc, candidates):
        """
        feature extraction is w.r.t each candidate .text is for reference.
        """
        # lets con
        # doc=doc.encode('ascii','ignore')

        # named_entities=self.get_named_entities(doc)
        # noun_phrases=self.get_nounphrases(doc)
        noun_adj_list = self.extract_noun_adjective(doc)
        candidates = self.filter_tokens(candidates, noun_adj_list)
        # keep noun and adjectives only
        features = {}
        orth_feat = self.get_orthographic_features(candidates, doc)
        features.update(orth_feat)
        # lex_feat=self.get_lexical_features(candidates,text,named_entities,noun_phrases)
        # features.update(lex_feat)
        wiki_feats = self.get_wiki_features(candidates)
        for key, value in wiki_feats.items():
            try:
                features[key].update(value)
            except Exception as e:
                pass
        # features.update(wiki_feats)


        return features

    def create_and_transform_to_vec(self, train_feat):
        v = DictVectorizer(sparse=False)
        train_vec = v.fit_transform(train_feat)
        self.dict_v = v
        joblib.dump(self.dict_v, self.dict_v_path)
        return train_vec

    def transform_to_vec(self, train_feat):
        return self.dict_v.transform(train_feat)

    def extract_training_features(self, train_data_iterator):
        # get feature for postive tokens
        features_list = []
        labels_list = []
        file_id = 0
        for (doc, keys) in train_data_iterator:
            # lets consider only single word tokens
            print('file id is.... ' + str(file_id))
            file_id += 1
            # we are doing for single keyword
            candidates = [key for key in keys if len(re.split('\s+', key)) == 1]
            candidate_features = self.extract_features_from_text(doc, candidates)
            features_list.extend(list(candidate_features.values()))
            labels_list.extend([1] * len(candidate_features))
            print('feature creation done')
            token_list = re.split('\W+', doc)
            invalid_tokens = [token for token in token_list if token not in candidates if len(token) >= 3]
            negative_token_features = self.extract_features_from_text(doc, invalid_tokens)
            features_list.extend(list(negative_token_features.values()))
            labels_list.extend([-1] * len(negative_token_features))
            print('invalid tokens features created')
        train_vec = self.create_and_transform_to_vec(features_list)
        scikit_save_model(zip(train_vec, labels_list), self.vec_file_loc)
        print('dump over ')

    def predict(self, doc, candidates):

        candidate_features = self.extract_features_from_text(doc, candidates)
        vec = self.transform_to_vec(list(candidate_features.values()))
        result_prob = self.model.predict_proba(vec)
        results = self.model.predict(vec)
        print('all done')

    def predict_iterator(self, test_data_iterator):
        self.model = scikit_load_model(self.model_path)
        self.dict_v = scikit_load_model(self.dict_v_path)
        for (doc, keys) in test_data_iterator:
            keys = [key for key in keys if len(re.split('\s+', key)) == 1]
            self.predict(doc, keys)
            token_list = re.split('\W+', doc)
            invalid_tokens = [token for token in token_list if token not in keys if len(token) >= 3]
            self.predict(doc, invalid_tokens)

    def train(self, train_data_iterator):
        # get feature for postive tokens
        # load=True
        # if not load:
        self.extract_training_features(train_data_iterator)

        train_vec, labels_list = zip(*scikit_load_model(self.vec_file_loc))
        param_grid = {}
        est = RandomForestClassifier(n_estimators=10)
        mdl = GridSearchCV(est, param_grid, n_jobs=1).fit(train_vec, labels_list)
        print('best params are')
        print(mdl.best_params_)
        self.model = mdl.best_estimator_
        print('best scores are ' + str(mdl.best_score_))
        joblib.dump(self.model, self.model_path, compress=3)
        # get feature for negative tokens
        pass


class CorpusIterator():
    def __init__(self, text_iter):
        self.text_iter = text_iter

    def __iter__(self):
        from nltk.corpus import stopwords
        stoplist = list(set(stopwords.words('english')))
        for (text, keys) in self.text_iter:
            texts = [word.lower() for word in re.split('\W+|\d+|_+', text) if word not in stoplist and len(word) > 1]
            yield texts


class TransformedIterator():
    def __init__(self, text_iter, dictionary):
        self.text_iter = text_iter
        self.dictionary = dictionary

    def __iter__(self):
        from nltk.corpus import stopwords
        stoplist = list(set(stopwords.words('english')))
        for (text, keys) in self.text_iter:
            texts = [word.lower() for word in re.split('\W+|\d+|_+', text) if word not in stoplist and len(word) > 1]
            yield self.dictionary.doc2bow(texts)


def build_tfidf_model():
    """
    in main code we will be resusing this model
    """
    from keyword_machine.training_data import CrowdDataIterator, TrainDataIterator

    from keyword_machine.dataset_citeulike_extractor import citeulike_iterator
    print("started tfidf_model")
    # ----------------------------------------------------------------------
    crowd_data_iter = CrowdDataIterator("500N-KPCrowd-v1.1/CorpusAndCrowdsourcingAnnotations/train/")
    citeulike180 = citeulike_iterator("citeulike180", "documents", "taggers")
    fao30 = citeulike_iterator("fao30", "documents", "indexers")
    fa0780 = CrowdDataIterator("./keyword-extraction-datasets-master/fao780/")
    wiki20 = citeulike_iterator("wiki20", "documents", "teams")
    data_set_appends = [crowd_data_iter, citeulike180, fao30, fa0780, wiki20]

    train_iter = TrainDataIterator(data_set_appends)
    # ----------------------------------------------------------------------



    # crowd_data_iter=CrowdDataIterator()
    # train_iter=TrainDataIterator([crowd_data_iter])

    corpus_iter = CorpusIterator(train_iter)
    dictionary = corpora.Dictionary(corpus_iter)
    trans_corpus = TransformedIterator(train_iter, dictionary)
    tfidf = models.TfidfModel(trans_corpus, normalize=True)
    # corpus_tfidf = tfidf[corpus]
    print("saving tfidf_model")
    scikit_save_model(tfidf, TFIDF_MDL_PATH)
    scikit_save_model(dictionary, DICT_PATH)


def build_classifier():
    from keyword_machine.training_data import CrowdDataIterator, TrainDataIterator
    from keyword_machine.dataset_citeulike_extractor import citeulike_iterator
    # crowd_data_iter=CrowdDataIterator()
    # train_iter = TrainDataIterator([crowd_data_iter])
    # -------------------------------------------------
    crowd_data_iter = CrowdDataIterator("500N-KPCrowd-v1.1/CorpusAndCrowdsourcingAnnotations/train/")
    citeulike180 = citeulike_iterator("citeulike180", "documents", "taggers")
    fao30 = citeulike_iterator("fao30", "documents", "indexers")
    fa0780 = CrowdDataIterator("./keyword-extraction-datasets-master/fao780/")
    wiki20 = citeulike_iterator("wiki20", "documents", "teams")
    data_set_appends = [crowd_data_iter, citeulike180, fao30, fa0780, wiki20]

    train_iter = TrainDataIterator(data_set_appends)
    # -------------------------------------------------

    key_word_clf = KeyWordClassifier(TFIDF_MDL_PATH, DICT_PATH, NER_TAGGER_MODEL_LOC, NER_TAGGER_MODEL_LOC,
                                     CLF_DICT_PATH, CLF_MDL_PATH)
    key_word_clf.train(train_iter)


def predict_classifier():
    from keyword_machine.training_data import CrowdDataIterator, TrainDataIterator
    import time
    from keyword_machine.dataset_citeulike_extractor import Doc_iterator
    doc_reader = Doc_iterator("wiki20", "documents")
    # crowd_data_iter=CrowdDataIterator()
    test_iter = TrainDataIterator([doc_reader])
    key_word_clf = KeyWordClassifier(TFIDF_MDL_PATH, DICT_PATH, NER_TAGGER_MODEL_LOC, NER_TAGGER_MODEL_LOC,
                                     CLF_DICT_PATH, CLF_MDL_PATH)
    t0 = time.time()
    key_word_clf.predict_iterator(test_iter)
    t1 = time.time()
    print('time taken is ' + str(t1 - t0))


# build_tfidf_model()
build_classifier()
# predict_classifier()












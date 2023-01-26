import pickle

import numpy as np
from graphtext import GraphText


class KeywordExtractor:
    """Extract keywords from text"""
    def __init__(self, damping_factor=0.85, convergence_threshold=1e-5, iteration_steps=50, processed_text=None):
        self.damping_factor = damping_factor  # usually  .85
        self.threshold = convergence_threshold
        self.iterations = iteration_steps
        self.processed_text = processed_text

    """Creates the graph for the algorithm"""
    def create_graph(self):

        """
        # Builds the graph
        graphtext = GraphText(self.processed_text.vocabulary, self.processed_text.processed_text)
        graphtext.create_graph()

        # Saves the graph in memory
        with open('graph0.pickle', 'wb') as handle:
            pickle.dump(graphtext, handle, protocol=pickle.HIGHEST_PROTOCOL)
        """
        with open('graph0.pickle', 'rb') as handle:
          graphtext = pickle.load(handle)

        return graphtext

    """Gives a score for every word which has been found in the vocabulary"""
    def score_vocabulary_words(self, graphtext):
        vocab_len = len(self.processed_text.vocabulary)

        # Gets the sum of weights of every vertex
        inout = np.zeros(vocab_len, dtype='f')
        for i in range(0, vocab_len):
            for j in range(0, vocab_len):
                inout[i] += graphtext.weighted_edge[i][j]

        for iteration in range(0, self.iterations):
            prev_score = np.copy(graphtext.score)
            for i in range(0, vocab_len):
                summation = 0

                for j in range(0, vocab_len):
                    if graphtext.weighted_edge[i][j] != 0:
                        summation += (graphtext.weighted_edge[i][j] / inout[j]) * graphtext.score[j]
                graphtext.score[i] = (1 - self.damping_factor) + self.damping_factor * summation

            if np.sum(np.fabs(prev_score - graphtext.score)) <= self.threshold:  # convergence condition
                print("Converging at iteration " + str(iteration) + "...")
                break

        print("Scores for single words done")

    """Gets keyphrases from the text"""
    def get_candidate_keyphrases(self):
        candidate_phrases = []
        candidate_phrase = " "
        for word in self.processed_text.lemmatized_text:
            if word in self.processed_text.stopwords:
                if candidate_phrase != " ":
                    candidate_phrases.append(str(candidate_phrase).strip().split())
                candidate_phrase = " "
            elif word not in self.processed_text.stopwords:
                candidate_phrase += str(word)
                candidate_phrase += " "

        unique_phrases = []
        # Deletes repeated ones
        for phrase in candidate_phrases:
            if phrase not in unique_phrases:
                unique_phrases.append(phrase)

        for word in self.processed_text.vocabulary:
            for phrase in unique_phrases:
                if (word in phrase) and ([word] in unique_phrases) and (len(phrase) > 1):
                    # if len(phrase)>1 then the current phrase is multi-worded.
                    # if the word in vocabulary is present in unique_phrases as a single-word-phrase
                    # and at the same time present as a word within a multi-worded phrase,
                    # then I will remove the single-word-phrase from the list.
                    unique_phrases.remove([word])

        # print(unique_phrases)
        return unique_phrases

    def score_keyphrases(self, phrases, graphtext):
        phrase_scores = []
        keywords = []
        for phrase in phrases:
            phrase_score = 0
            keyword = ''
            for word in phrase:
                keyword += str(word)
                keyword += " "
                if word in self.processed_text.vocabulary:
                    phrase_score += graphtext.score[self.processed_text.vocabulary.index(word)]
            phrase_scores.append(phrase_score)
            keywords.append(keyword.strip())
        i = 0
        for keyword in keywords:
            print("Keyword: '" + str(keyword) + "', Score: " + str(phrase_scores[i]))
            i += 1
        return keywords, phrase_scores

    """Returns the keywords of the text"""
    def get_keywords(self, text):
        # Creates the graph for the text and execute the algorithm
        graphtext = self.create_graph()

        # calculates the punctuation
        self.score_vocabulary_words(graphtext)

        # gets candidate phrases
        phrases = self.get_candidate_keyphrases()

        # scores all the phrases
        keywords, phrase_scores = self.score_keyphrases(phrases, graphtext)

        # Get ranked keywords
        sorted_index = np.flip(np.argsort(phrase_scores), 0)
        keywords_num = 10
        print("Keywords found:\n")
        for i in range(0, keywords_num):
            print(str(keywords[sorted_index[i]]))



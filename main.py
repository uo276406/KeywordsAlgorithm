from graph.graph import Graph
from keywordextractor import KeywordExtractor
from processor.textprocessor import TextProcessor
import pickle


def main():
    for i in range(0, 10):
        f = open("./data/History5/dataSet/From yesterday to tomorrow _ history and citizenship education_glossary_" + str(i) + ".txt", 'r', encoding='utf8')
        text = f.read()
        f.close()
        # text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."
        # nltk.download('averaged_perceptron_tagger')
        # nltk.download('wordnet')

        # Process the text
        processor = TextProcessor(text)
        processor.process_text()

        # ----------------------------------------------------------------------
        # Creates the graph

        graph = Graph(processor.vocabulary, processor.filtered_text)
        graph.create_graph()
        # Saves the graph in memory
        with open('./graphsaved/History5/From yesterday to tomorrow _ history and citizenship education_glossary_' + str(i) + '.pickle', 'wb') as handle:
            pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
    exit()

    """
    # ----------------------------------------------------------------------
    # Loads the graph from memory
    graph = None
    with open('./graphsaved/graph' + str(i) + '.pickle', 'rb') as handle:
        graph = pickle.load(handle)
    # ----------------------------------------------------------------------

    # Looks for the keywords
    extractor = KeywordExtractor(processor=processor, graph=graph, damping_factor=0.85,
                                 convergence_threshold=1e-2, iteration_steps=10)
    keywords_found = extractor.get_keywords()

    # Count the keywords found
    key = open("./data/History/dataSet/USHist_" + str(i) + ".key", 'r', encoding='utf8')
    keywords_real = key.readlines()
    key.close()

    print("Keywords found for USHist_" + str(i))
    print(keywords_found)
    print()
    """


if __name__ == "__main__":
    main()

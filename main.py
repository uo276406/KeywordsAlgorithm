from graph.graph import Graph
from keywordextractor import KeywordExtractor
from processor.textprocessor import TextProcessor
import pickle
from nltk.stem import WordNetLemmatizer
from spacy.lang.en.stop_words import STOP_WORDS
import xlsxwriter

data = [['USHist_', 32], ['chapter', 37], ['Cambridge_IGCSE_History_', 10], ['From yesterday to tomorrow _ history and citizenship education_glossary_', 6]]


def write_row_excel(worksheet, chapter, tp, fn, fp, precision, recall, f1score, row_num):
    worksheet.write(row_num+1, 0, chapter)
    worksheet.write(row_num+1, 1, tp)
    worksheet.write(row_num+1, 2, fp)
    worksheet.write(row_num+1, 3, fn)
    worksheet.write(row_num+1, 4, precision)
    worksheet.write(row_num+1, 5, recall)
    worksheet.write(row_num+1, 6, f1score)

def main():
    wordnet_lemmatizer = WordNetLemmatizer()

    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook('./results.xlsx')

    for name, iterations in data:

        worksheet = workbook.add_worksheet(name=name)
        worksheet.write(0, 0, "text name")
        worksheet.write(0, 1, "tp")
        worksheet.write(0, 2, "fp")
        worksheet.write(0, 3, "fn")
        worksheet.write(0, 4, "precision")
        worksheet.write(0, 5, "recall")
        worksheet.write(0, 6, "f1score")

        for i in range(iterations):
            f = open("./data/History/dataSet/" + name + str(i) + ".txt", 'r', encoding='utf8')
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
            """
            graph = Graph(processor.vocabulary, processor.filtered_text)
            graph.create_graph()
            # Saves the graph in memory
            with open('./graphsaved/History/'+ name + str(i) + '.pickle', 'wb') as handle:
                pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
            """
            # ----------------------------------------------------------------------
            # Loads the graph from memory
            graph = None
            with open('./graphsaved/History/' + name + str(i) + '.pickle', 'rb') as handle:
                graph = pickle.load(handle)
            # ----------------------------------------------------------------------

            # Looks for the keywords
            extractor = KeywordExtractor(processor=processor, graph=graph, damping_factor=0.85,
                                         convergence_threshold=1e-3, iteration_steps=15)
            keywords_found = extractor.get_keywords()

            # Count the keywords found
            key = open("./data/History/dataSet/" + name + str(i) + ".key", 'r', encoding='utf8')
            keywords_real = key.readlines()
            key.close()

            tp = 0
            for k in keywords_real:
                keyword = k.lower().strip()
                keyword_processed = keyword.split()
                keyword_to_check = ""
                for elem in keyword_processed:
                    if elem not in STOP_WORDS:
                        keyword_to_check += (elem + " ")
                if str(wordnet_lemmatizer.lemmatize(keyword_to_check.strip())) in keywords_found:
                    tp += 1

            fn = len(keywords_real)-tp
            fp = len(keywords_found)-tp
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1score = (2 * tp) / (2 * tp + fp + fn)

            print("Keywords found for " + name + str(i))
            print(keywords_found)
            print("TP:" + str(tp))
            print("FN:" + str(fn))
            print("FP:" + str(fp))
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("F1Score: " + str(f1score))
            print()
            write_row_excel(worksheet, name + str(i), tp, fn, fp, precision, recall, f1score, i)

    workbook.close()


if __name__ == "__main__":
    main()

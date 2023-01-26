from keywordextractor import KeywordExtractor
from textprocessor import TextProcessor


f = open("./data/History/dataSet/USHist_0.txt", 'r', encoding='utf8')
text = f.read()
f.close()

# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

# Process the text
processed_text = TextProcessor(text)
processed_text.process_text()

extractor = KeywordExtractor(processed_text=processed_text)
extractor.get_keywords(text)

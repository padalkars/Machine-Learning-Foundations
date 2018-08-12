from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from stem.itrstem import IterativeStemmer
from tokenize.tokenizer import Tokenizer
from rmvstopwords import StopWordRemover


stemmer = IterativeStemmer()

d1 = "त्यो घर रातो छ"
d2 = "यो निलो कलम हो"
d3 = "भाईको घरमा हो"
documents = [d1, d2, d3]

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(w) for w in analyzer(doc))

print("------------------- Method-I -------------------")
vectorizer = StemmedCountVectorizer(stop_words=StopWordRemover().get_stopwords(),
                                    tokenizer=lambda text: Tokenizer().word_tokenize(text=text), analyzer='word')
tf_matrix = vectorizer.fit_transform(documents)
print("========= Vocabulary =========")
print(vectorizer.vocabulary_)

print(" ========= Term Frequency ===========")
print(tf_matrix.todense())
print()

tfidf = TfidfTransformer(norm="l2")
tfidf.fit(tf_matrix)

print("========= Inverse Document Frequency =========")
print(tfidf.idf_)
print()

tf_idf_matrix = tfidf.transform(tf_matrix)
print("========= TF-IDF Matrix =========")
print(tf_idf_matrix.todense())
print()





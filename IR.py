from flask import Flask , render_template , redirect, request, url_for
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel,cosine_similarity
from nltk.stem.snowball import SnowballStemmer

app = Flask(__name__)

def pre_process(corpus):
	corpus_new=[]
	for text in corpus:
		# lower 
		text = text.lower()
		#remove tags
		text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", text)
		# remove special characters and digits
		text = re.sub("\\d|\\W+"," ",text)
		corpus_new.append(text)
	return corpus_new
def get_stop_words(stop_file_path):
    """load stop words """   
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)

def find_similar(a, index, top_n = 1400):
	cosine_similarities = linear_kernel(a, a[-1:]).flatten()
	related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
	return [(index, cosine_similarities[index]) for index in related_docs_indices if cosine_similarities[index]!=0.0][1:top_n]

@app.route('/')
def index():
	return render_template("index.html")
@app.route('/search', methods=['POST','GET'])
def IR():
	if request.method=='POST':
		query=request.form['query']
		corpus=[]
		for file in os.listdir('document'):
			if file.endswith('.txt'):
				doc = open('document/'+file,'r').read()
			corpus.append(doc)
		corpus.append(query)
		corpus_new=pre_process(corpus)
		stopwords=get_stop_words("stopwords.txt")
		vectorizer=CountVectorizer(max_df=0.85,min_df=1,stop_words=stopwords,analyzer='word', max_features=None)
		word_count_vector =vectorizer.fit_transform(corpus_new).toarray()
		transformer=TfidfTransformer(smooth_idf=True,use_idf=True).fit(word_count_vector)
		transformer.fit(word_count_vector)
		a=transformer.transform(word_count_vector).toarray()
		results=find_similar(a,len(corpus_new))
		result_number=len(results)
		return render_template("IR.html",results=results,corpus_new=corpus_new,r=result_number)

@app.route('/search/<int:id>')
def show_id(id):
	corpus=[]
	for file in os.listdir('document'):
		if file.endswith('.txt'):
			doc = open('document/'+file,'r').read()
		corpus.append(doc)
	return render_template("showID.html", corpus=corpus, id=id)
if __name__ == '__main__':
    app.run(debug=True)

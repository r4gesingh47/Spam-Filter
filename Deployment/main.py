import flask
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


app = flask.Flask(__name__, template_folder='templates')

with open(f'models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(f'models/feature.pkl', 'rb') as f:
    feature = pickle.load(f)

tfid=TfidfVectorizer(analyzer='char', ngram_range=(2,3),vocabulary=feature.vocabulary_)

@app.route('/',methods=['GET', 'POST'])

def main():
	def clean_input(s):
	    s=re.sub(r'[^\w\s]', '',s)
	    stop_words=set(stopwords.words('english'))
	    tokens= word_tokenize(s)
	    cleaned = [word for word in tokens if word not in stop_words]
	    return " ".join(cleaned)
	if flask.request.method=='GET':
		return flask.render_template('main.html')
	if flask.request.method=='POST':
		user_input=flask.request.form['TextBox']
		input_cleaned=[clean_input(user_input)]
		value=tfid.fit_transform(input_cleaned)
		model_input=value
		result_pred="Failed"
		predicted_val=model.predict(model_input)[0]
		if predicted_val==0:
			result_pred='Ham'
		if predicted_val==1:
			result_pred='Spam'

		return flask.render_template('main.html',email=user_input,result=result_pred)



if __name__=='__main__':
	app.run()
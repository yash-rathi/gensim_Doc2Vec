from gensim import models, similarities
from os import listdir
from os.path import isfile, join
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import gensim

docLabels = []
docLabels = [f for f in listdir("csv/") if f.endswith('.csv')]

data = []
for doc in docLabels:
    data.append(open("csv/" + doc, 'r'))

tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))

def nlp_clean(data):
   new_data = []
   for d in data:
      d = str(d)
      new_str = d.lower()
      dlist = tokenizer.tokenize(new_str)
      dlist = list(set(dlist).difference(stopword_set))
      new_data.append(dlist)
   return new_data

class LabeledLineSentence(object):
	def __init__(self, doc_list, labels_list):
		self.labels_list = labels_list
		self.doc_list = doc_list
	def __iter__(self):
#		for idx, doc in enumerate(self.doc_list):
#			yield gensim.models.doc2vec.LabeledSentence(doc,[self.labels_list[idx]])
		for uid, line in enumerate(self.doc_list):
			yield gensim.models.doc2vec.LabeledSentence(line,['SENT_%s' % uid])

data = nlp_clean(data)
it = LabeledLineSentence(data, docLabels)

model = gensim.models.Doc2Vec(size=32, min_count=0, alpha=0.025, min_alpha=0.025)
model.build_vocab(it)

for epoch in range(100):
	#print ('iteration '+str(epoch+1))
	model.train(it)
	model.alpha -= 0.002
	model.min_alpha = model.alpha
	model.train(it)

'''
model.save(‘doc2vec.model’)
print “model saved”
'''

docvec = model.docvecs[0]
print (docvec)
print ("\n\n\n")

print (model.docvecs.most_similar(['SENT_0']))
'''
#printing the vector of the file using its name
docvec = model.docvecs['1.csv']
print (docvec)
'''

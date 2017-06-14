from gensim import models, similarities
from os import listdir
from os.path import isfile, join
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import gensim
import pandas as pd
from gensim.models import Doc2Vec
import time

docLabels = []
docLabels = [f for f in listdir("csv/") if f.endswith('.txt')]

data = []
for doc in docLabels:
    data.append(open("csv/" + doc, 'r'))
'''
    fd = pd.read_csv("csv/" + doc)
    ques = fd['Questions'].values.tolist()
    ans = fd['Answers'].values.tolist()
    corpus = ques + ans
    data.append(corpus)

def nlp_clean(data):
   new_data = []
   for d in data:
      d = str(d)
      new_data.append(d)
   return new_data
'''
no_line = 0
class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = doc
        self.lables = dict()
#        self.id = dict()
    def __iter__(self):
        for uid, line in enumerate(open("csv/"+doc)):
            self.lables['SENT_%s' % uid]=line
#            self.id = uid
#            no_line = no_line + 1
            yield gensim.models.doc2vec.LabeledSentence(line.strip().split(), ['SENT_%s' % uid])

#data = nlp_clean(data)
doc.split()
it = LabeledLineSentence(docLabels)

def train1(it1):
	model = gensim.models.Doc2Vec(size=200, min_count=3, alpha=0.025, min_alpha=0.025, iter=20)
	model.build_vocab(it1) 
	for epoch in range(100):
		model.train(it)
		model.alpha -= 0.002
		model.min_alpha = model.alpha
		model.train(it)
	model.save("d2v.model")

start_time = time.time()
train1(it)
model=Doc2Vec.load("d2v.model") 

docvec = model.docvecs[10]
#print (docvec)
#print ("\n\n\n")
#print ("Similart to:- "+it.lables['SENT_0'])
#print ("\n")
#for w,v in sorted(model.docvecs.most_similar(['SENT_0'], topn=2), key=lambda x: x[1], reverse=True): print(it.lables[w]+":"+str(v))
#print ("\n\n")

f = open('out.txt', 'w')
for i in range(883):
	f.write(str(it.lables['SENT_%s' %i])+" "+str(model.docvecs[i]))
	f.write("\n\n")

f.close()
print("--- %s seconds ---" % (time.time() - start_time))

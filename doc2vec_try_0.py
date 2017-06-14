'''
from os import listdir
from os.path import isfile, join

docLabels = []
docLabels = [f for f in listdir("Downloads/") if f.endswith('.csv')]

data = []
for doc in docLabels:
    data.append(open("Downloads/" + doc, 'r')
'''

from gensim import models, similarities

sentence = models.doc2vec.LabeledSentence(
    words=[u'some', u'words', u'here'], tags=['SENT_0'])
sentence1 = models.doc2vec.LabeledSentence(
    words=[u'here', u'we', u'go'], tags=['SENT_1'])

sentences = [sentence, sentence1]

#class LabeledLineSentence(object):
#    def __init__(self, filename):
#        self.filename = filename
#    def __iter__(self):
#        for uid, line in enumerate(open(filename)):
#            yield LabeledSentence(words=line.split(), labels=['SENT_%s' % uid])
            
model = models.Doc2Vec(alpha=.025, min_alpha=.025, min_count=1)
model.build_vocab(sentences)

for epoch in range(10):
    model.train(sentences)
    model.alpha -= 0.002
    model.min_alpha = model.alpha

#model.save("my_model.doc2vec")
#model_loaded = models.Doc2Vec.load('my_model.doc2vec')

#print (model.most_similar('SENT_0'))
print (model.docvecs.most_similar(['SENT_0']))
print (model.docvecs.most_similar("SENT_1"))

import twokenize
import codecs
import matplotlib.pyplot as plt
import math
import numpy as np

stopf = codecs.open('stopwords.txt', 'rU', 'utf-8')
stopwords = [line.lower() for line in stopf]

token2idmap = {}
id2tokenmap = {}
#dict = {}
dictsizes = [100, 1000, 5000]
dictofdict = {}

def GetTokenId(token):
  if token not in token2idmap:
    tokenid = len(token2idmap) + 1
    token2idmap[token] = tokenid
    id2tokenmap[tokenid] = token
  
  return token2idmap[token]
    
def TokenizeTweet(tweet):
  vec = [item.lower() for item in twokenize.simple_tokenize(tweet)]
  outputvec = [GetTokenId(item) for item in vec if not item.startswith('http') and not item.startswith('@') and item not in stopwords and item != '#tcot' and item != '#p2' and len(item) > 1]
  return outputvec
  
def CreateSizedDictionaries(pmitcot, pmip2):
  for dictsize in dictsizes:
    tmpdict = {}
    lentcot = len(pmitcot)
    lenp2 = len(pmip2)
    i = 0
    j = 0
    while i < lentcot and j < lenp2 and len(tmpdict) < dictsize:
      if pmitcot[i][0] > 0 and pmitcot[i][0] not in tmpdict: tmpdict[pmitcot[i][0]] = pmitcot[i][1]
      if pmip2[j][0] > 0 and pmip2[j][0] not in tmpdict: tmpdict[pmip2[j][0]] = pmip2[j][1]
      i += 1
      j += 1
    
    while i < lentcot and len(tmpdict) < dictsize:
      if pmitcot[i][0] > 0 and pmitcot[i][0] not in tmpdict: tmpdict[pmitcot[i][0]] = pmitcot[i][1]
      i += 1

    while j < lenp2 and len(tmpdict) < dictsize:
      if pmip2[j][0] > 0 and pmip2[j][0] not in tmpdict: tmpdict[pmip2[j][0]] = pmip2[j][1]
      j += 1
      
    if dictsize != len(tmpdict): print 'dictionary size:%d is not equal to expected vocabulary size %d'%(len(tmpdict), dictsize)
    dictofdict[dictsize] = tmpdict
  
def CreateFileSplits():
  for dictsize in dictsizes:
    tmpdict = dictofdict[dictsize]
    vec = [ tuple for tuple in sorted(tmpdict.items(), reverse = True, key = lambda x: x[1]) ]
    
    dictf = open('dict_'+str(dictsize)+'.txt', 'w')
    for tuple in vec:dictf.write(str(tuple[0])+'\n')
    dictf.close()
    
    files = ['trainTCOT.txt', 'testTCOT.txt', 'trainP2.txt', 'testP2.txt']
    for file in files:
      ipf = codecs.open(file, 'rU', 'utf-8')     
      opf = open(file[:-4] + '_' + str(dictsize) + '.txt', 'w')
      
      for tweet in ipf:
        counts = {}
        tmpvec = TokenizeTweet(tweet)
        for itemid in tmpvec:
          if itemid in tmpdict:
            if itemid in counts: counts[itemid] = counts[itemid] + 1
            else: counts[itemid] = 1
        
        opf.write('1')
        for tuple in vec:
          val = 0
          if tuple[0] in counts: val = counts[tuple[0]]
          opf.write(' '+str(val))
          
        opf.write('\n')
      
      opf.close()
      ipf.close()
    
def GetPMI():

  lineno = 0
  wordlistf = codecs.open('wordlist.txt', 'rU', 'utf-8')
  for line in wordlistf:
    lineno = lineno + 1
    token = line
    if len(line) >= 1 and  line[-1] == '\n': token = line[:-1]
    if len(line) >=2 and line[-2] == '\r': token = line[:-2]
    
    token2idmap[token] = lineno
    id2tokenmap[lineno] = token
  
  traintcot = np.zeros(lineno+1, dtype = int)
  trainp2 = np.zeros(lineno+1, dtype = int)
  
  tcotf = codecs.open('trainTCOT.txt', 'rU', 'utf-8')
  for tweet in tcotf:
    vec = TokenizeTweet(tweet)
    for itemid in vec: traintcot[itemid] += 1
  tcotf.close()
  
  p2f = codecs.open('trainP2.txt', 'rU', 'utf-8')
  for tweet in p2f:
    vec = TokenizeTweet(tweet)
    for itemid in vec: trainp2[itemid] += 1
  p2f.close()

  pwgiventcot = traintcot / float(np.sum(traintcot))
  pwgivenp2 = trainp2 / float(np.sum(trainp2))
  
  pw = pwgiventcot + pwgivenp2
  
  pmitcot = pwgiventcot / pw
  pmip2 = pwgivenp2 / pw
  
  tcotvec = [ (k, pmitcot[k]) for k in range(lineno+1) ]
  tcotp2 = [ (k, pmip2[k]) for k in range(lineno+1) ]
  
  top10tcot = sorted(tcotvec, key = lambda x : x[1], reverse = True)
  top10p2 = sorted(tcotp2, key = lambda x: x[1], reverse = True)

  opf = codecs.open('pmitcot.txt', 'w', 'utf-8')
  for tuple in top10tcot:
    if tuple[0] > 0: opf.write(id2tokenmap[tuple[0]] + ' pmi:'+ str(tuple[1]) + '\n')
    else: opf.write('OFFSET_TERM pmi:'+str(tuple[1]) + '\n')
  opf.close()

  opf = codecs.open('pmip2.txt', 'w', 'utf-8')
  for tuple in top10p2:
    if tuple[0] > 0: opf.write(id2tokenmap[tuple[0]] + ' pmi:'+ str(tuple[1]) + '\n')
    else: opf.write('OFFSET_TERM pmi:'+str(tuple[1]) + '\n')
  opf.close()
  
  return (top10tcot, top10p2)
  
def main():
  (pmitcot, pmip2) = GetPMI()
  
  CreateSizedDictionaries(pmitcot, pmip2)
  CreateFileSplits()

if __name__=='__main__':
  main()


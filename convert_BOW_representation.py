import twokenize
import codecs
import matplotlib.pyplot as plt
import math

stopf = codecs.open('stopwords.txt', 'rU', 'utf-8')
stopwords = [line.lower() for line in stopf]

token2idmap = {}
id2tokenmap = {}
dict = {}
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
  
def ProcessTweet(tweet):
  vec = TokenizeTweet(tweet)
  for itemid in vec:
    if itemid not in dict: dict[itemid] = 1
    else: dict[itemid] = dict[itemid] + 1

def WriteId2TokenMappingToFile(): #NOTE: tokens are just written to file assuming a 1-based index. while reading them, read them using a 1-based index
  wordlistf = codecs.open('wordlist.txt', 'w', 'utf-8')
  for id in range(1, len(id2tokenmap)+1): wordlistf.write(id2tokenmap[id] + '\n')
  wordlistf.close()
  
def Deliverable3a():
  vecx = [ math.log(item) for item in range(1, len(dict)+1) ]
  vecy = [ math.log(item) for item in sorted(dict.values(), reverse = True)]

  plt.clf()
  plt.plot(vecx,vecy)
  plt.title('zipf law for Good/Bad Customer Service training data')
  plt.ylabel('log(count)')
  plt.xlabel('log(rank)')
  plt.savefig('zipf.pdf',format='pdf')
  
def Deliverable3b():
  f3b = codecs.open('Deliverable3bOutput.txt', 'w', 'utf-8')
  
  f3b.write('Size of Full Training Dictionary:' + str(len(dict)) + '\n')
  for i  in range(80): f3b.write('-');
  f3b.write('\n')
  
  for dictsize in dictsizes:
    f3b.write('Dictionary Size:' + str(dictsize) + '\n')
    vec = [ tuple for tuple in sorted(dict.items(), reverse = True, key = lambda x: x[1])[:dictsize] ]
    for tuple in vec[-5:]:
      f3b.write(id2tokenmap[tuple[0]] + u' Count:' + str(tuple[1]).encode('utf-8') + u'\n')
    
    for i  in range(80): f3b.write('-');
    f3b.write('\n')
    
    tmpdict = {}
    for tuple in vec: tmpdict[tuple[0]] = tuple[1]
    dictofdict[dictsize] = tmpdict
    
  f3b.close()
  
  
def Deliverable3c():
  for dictsize in dictsizes:
    tmpdict = dictofdict[dictsize]
    vec = [ tuple for tuple in sorted(tmpdict.items(), reverse = True, key = lambda x: x[1]) ]
    
    dictf = open('dict_'+str(dictsize)+'.txt', 'w')
    for tuple in vec:dictf.write(str(tuple[0])+'\n')
    dictf.close()
    
    files = ['traingoodcustserv.txt', 'testgoodcustserv.txt', 'trainbadcustserv.txt', 'testbadcustserv.txt']
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
    

def main():
  tcotf = codecs.open('traingoodcustserv.txt', 'rU', 'utf-8')
  p2f = codecs.open('trainbadcustserv.txt', 'rU', 'utf-8')

  for tweet in tcotf: ProcessTweet(tweet)
  for tweet in p2f: ProcessTweet(tweet)

  stopf.close()
  p2f.close()
  tcotf.close()

  WriteId2TokenMappingToFile()
  Deliverable3a()
  Deliverable3b()
  Deliverable3c()

if __name__=='__main__':
  main()


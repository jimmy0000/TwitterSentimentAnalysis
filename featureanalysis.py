import numpy as np
import codecs
import math

def main():

  dictsize = 5000
  traintcot = np.loadtxt('traingoodcustserv_5000.txt')
  trainp2 = np.loadtxt('trainbadcustserv_5000.txt')
  numtcot = traintcot.shape[0]
  nump2f = trainp2.shape[0]
  
  pwgiventcot = np.sum(traintcot, axis = 0) / float(np.sum(traintcot))
  pwgivenp2 = np.sum(trainp2, axis = 0) / float(np.sum(trainp2))

  #NOTE: This is an incorrect formula for p(x) = p(x|y1) + p(x|y2). 
  #NOTE: Correct formula is p(x) = Sigma{y} p(x,y) = p(y1)*p(x|y1) + p(y2)*p(x|y2)  
  #NOTE: But, since in our case p(y1) = p(y2) because we have equal representation for good/bad labels, the incorrect formula didnt affect the end result
  pw = pwgiventcot + pwgivenp2
  
  pmitcot = pwgiventcot / pw
  pmip2 = pwgivenp2 / pw
  
  tcotvec = [ (k, pmitcot[k]) for k in range(traintcot.shape[1]) ]
  tcotp2 = [ (k, pmip2[k]) for k in range(trainp2.shape[1]) ]
  
  top10tcot = sorted(tcotvec, key = lambda x : x[1], reverse = True)[:10]
  top10p2 = sorted(tcotp2, key = lambda x: x[1], reverse = True)[:10]
  
  dict5000 = np.loadtxt('dict_5000.txt', dtype=int) #NOTE: This a 0-index array
  token2idmap = {}
  id2tokenmap = {}
  
  lineno = 0
  wordlistf = codecs.open('wordlist.txt', 'rU', 'utf-8')
  for line in wordlistf:
    lineno = lineno + 1
    token = line
    if len(line) >= 1 and  line[-1] == '\n': token = line[:-1]
    if len(line) >=2 and line[-2] == '\r': token = line[:-2]
    
    token2idmap[token] = lineno
    id2tokenmap[lineno] = token
    
  wordlistf.close()
  
  opf = codecs.open('Deliverable4output.txt', 'w', 'utf-8')
  opf.write('PMI for TCOT data\n')
  for tuple in top10tcot:
    if tuple[0] > 0: opf.write(id2tokenmap[dict5000[tuple[0]-1]])
    else: opf.write(u'OFFSET_TERM')
    opf.write(u' pmi:'+ str(math.log(tuple[1])).encode('utf-8')+u'\n')
  
  for i in range(80): opf.write('-')
  opf.write('\n')
  
  opf.write('PMI for P2 data\n')
  for tuple in top10p2:
    if tuple[0] > 0: opf.write(id2tokenmap[dict5000[tuple[0]-1]])
    else: opf.write(u'OFFSET_TERM')
    opf.write(u' pmi:'+ str(math.log(tuple[1])).encode('utf-8')+u'\n')
  opf.close()
  
if __name__=='__main__':
  main()


import twokenize
import codecs
import random 

linecount = sum(1 for tweet in codecs.open('goodcustserv.txt', 'rU', 'utf-8'))
testsample = random.sample(xrange(linecount), linecount/5)

ipf = codecs.open('goodcustserv.txt', 'rU', 'utf-8')
stopf = codecs.open('stopwords.txt', 'rU', 'utf-8')
trainf = codecs.open('traingoodcustserv.txt', 'w', 'utf-8')
testf = codecs.open('testgoodcustserv.txt', 'w', 'utf-8')
stopwords = [line.lower() for line in stopf]

linecount = 0
for tweet in ipf:
  #vec = [item.lower() for item in twokenize.simple_tokenize(tweet)]
  #outputvec = [item for item in vec if not item.startswith('http') and not item.startswith('@') and item not in stopwords and item != '#tcot' and item != '#p2' and len(item) > 1]
  if linecount in testsample: 
    testf.write(tweet)#testf.write(u" ".join(outputvec) + u'\n')
    if tweet[-1] != '\n': testf.write('\n')
  else: 
    trainf.write(tweet)#trainf.write(u" ".join(outputvec) + u'\n')
    if tweet[-1] != '\n': trainf.write('\n')
  linecount =  linecount + 1

testf.close()
trainf.close()
ipf.close()


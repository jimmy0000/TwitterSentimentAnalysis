import twokenize
import codecs

ipf = codecs.open('deliverable2input.txt', 'rU', 'utf-8')
stopf = codecs.open('stopwords.txt', 'rU', 'utf-8')
opf = codecs.open('deliverable2output.txt', 'w', 'utf-8')
stopwords = [line.lower() for line in stopf]

for tweet in ipf:
  opf.write(tweet)
  if tweet[-1] != '\n': opf.write('\n');
  vec = [item.lower() for item in twokenize.simple_tokenize(tweet)]
  outputvec = [item for item in vec if not item.startswith('http') and not item.startswith('@') and item not in stopwords and item != '#tcot' and item != '#p2' and len(item) > 1]
  opf.write(u" ".join(outputvec) + u'\n')
  for i in range(80):opf.write('-')
  opf.write('\n')


opf.close()
ipf.close()
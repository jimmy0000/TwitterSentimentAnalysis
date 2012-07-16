import sys
import tweepy
import codecs

auth1 = tweepy.auth.OAuthHandler('Consumer Key','Consumer Secret') #Replace 'Consumer Key' and 'Consumer Secret' with required authorization keys
auth1.set_access_token('Access Token','Access Token Secret')	   #Key 'Access Token' and 'Access Token Secret' with required authorization keys
api = tweepy.API(auth1)
f=codecs.open('mydataset.txt', 'w', 'utf-8')

class StreamListener(tweepy.StreamListener):
    def on_status(self, status):
        try:
            tweet = status.text
            if not ('#p2' in tweet) :	#Exclude tweets with #p2 tags when we searching for tweets with #tcot tags, this gives us well-classified data
            	if not ('RT ' in tweet) :	#Exclude re-tweets
            		f.write(tweet.encode('utf-8') + '\n')
                f.flush()
            		#print tweet.encode('utf-8')#print "%s" % (tweet)
                
        except Exception, e:
            print >> sys.stderr, 'Encountered Exception:', e
            pass
    def on_error(self, status_code):
        print >> sys.stderr, 'Encountered error with status code:', status_code
        return True
    def on_timeout(self):
        print >> sys.stderr, 'Timeout...'
        return True
        
l = StreamListener()
streamer = tweepy.Stream(auth=auth1, listener=l)#, timeout=3000000000 )
#setTerms = ['#tcot']	#Can include a comma seperated list of any number of topics in which you are interested
setTerms = ['customer service']
streamer.filter(None,setTerms)


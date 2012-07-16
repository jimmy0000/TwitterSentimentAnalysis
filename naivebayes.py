import numpy as np
import random as rand
import matplotlib.pyplot as plt
import math

includeunigrams = 1
includebigrams = 1
skipwordlen = 2

def main():
  dictsizes = [100, 1000, 5000]
  EPSILON = 1e-9
  
  for dictsize in dictsizes:
    traintcot = np.loadtxt('trainTCOT' + '_' + str(dictsize) + '.txt', dtype = int) + EPSILON
    trainp2 = np.loadtxt('trainP2' + '_' + str(dictsize) + '.txt', dtype = int) + EPSILON
    testtcot = np.loadtxt('testTCOT' + '_' + str(dictsize) + '.txt', dtype = int) + EPSILON
    testp2 = np.loadtxt('testP2' + '_' + str(dictsize) + '.txt', dtype = int) + EPSILON

    numtraintcot = traintcot.shape[0]
    numtrainp2 = trainp2.shape[0]
    numtesttcot = testtcot.shape[0]
    numtestp2 = testp2.shape[0]
    numfeatures = traintcot.shape[1]
    if numfeatures != trainp2.shape[1]:
      print 'NumFeatures in TrainTCOT and TrainP2 not equal'
      return
    if numfeatures != testtcot.shape[1]:
      print 'NumFeatures in TrainTCOT and TestTCOT not equal'
      return
    if numfeatures != testp2.shape[1]:
      print 'NumFeatures in TrainTCOT and TestP2 not equal'
      return  
    
    Xtrain = np.zeros( (numtraintcot + numtrainp2, numfeatures) )
    Xtrain[:numtraintcot,] = traintcot
    Xtrain[numtraintcot:,] = trainp2
    Ytrain = np.zeros(numtraintcot + numtrainp2)
    Ytrain[:numtraintcot] = 1  
    Xtest = np.zeros( (numtesttcot + numtestp2, numfeatures) )
    Xtest[:numtesttcot,] = testtcot
    Xtest[numtesttcot:,] = testp2
    Ytest = np.zeros(numtesttcot + numtestp2)
    Ytest[:numtesttcot] = 1  

    smoothingfactorvec = [0, 0.1, 0.2, 0.5, 0.75, 1, 2, 10, 100]
    trainaccuracyvec = []
    testaccuracyvec = []

    for smoothingfactor in smoothingfactorvec:
      print smoothingfactor, numfeatures
      denominator = smoothingfactor
      numerator = 1
      if smoothingfactor <= 0: numerator = 0
      pwgiventcot = (np.sum(traintcot, axis = 0) + numerator*float(smoothingfactor)) / (float(np.sum(traintcot)) + denominator*float(numfeatures))
      pwgivenp2 = (np.sum(trainp2, axis = 0) + numerator*float(smoothingfactor)) / (float(np.sum(trainp2)) + denominator*float(numfeatures))

      pwgiventcot = np.log(pwgiventcot)
      pwgivenp2 = np.log(pwgivenp2)
      
      ptcot = float(numtraintcot) / (float(numtraintcot) + float(numtrainp2))
      pp2 = float(numtrainp2) / (float(numtraintcot) + float(numtrainp2))

      #Get accuracy rate on both training and test sets
      trainaccuracy = 0.0
      testaccuracy = 0.0
      for i in range(numtraintcot + numtrainp2):
        tcotscore = math.log(ptcot) + sum(np.multiply(pwgiventcot, Xtrain[i,:]))
        p2score = math.log(pp2) + sum(np.multiply(pwgivenp2, Xtrain[i,:]))
        yhat = 1
        if p2score > tcotscore + tcotscore * EPSILON: yhat = 0
        if yhat == Ytrain[i]: trainaccuracy = trainaccuracy + 1.0
      trainaccuracyvec.append(trainaccuracy / float(numtrainp2 + numtraintcot))        
        
      for i in range(numtesttcot + numtestp2):
        tcotscore = math.log(ptcot) + sum(np.multiply(pwgiventcot, Xtest[i,:]))
        p2score = math.log(pp2) + sum(np.multiply(pwgivenp2, Xtest[i,:]))
        yhat = 1
        if p2score > tcotscore + tcotscore * EPSILON: yhat = 0
        if yhat == Ytest[i]: testaccuracy = testaccuracy + 1.0
      testaccuracyvec.append(testaccuracy / float(numtestp2 + numtesttcot))        
      
    plt.clf()
    plt.plot(smoothingfactorvec, trainaccuracyvec)
    plt.plot(smoothingfactorvec, testaccuracyvec)
    plt.legend(['naivebayes-train', 'naivebayes-test'], loc=4)
    plt.title('naivebayes accuracy  on tcot/p2 for vocabulary size:'+str(dictsize))
    plt.ylabel('accuracy rate')
    plt.xlabel('smoothing factor')
    minval = min(min(trainaccuracyvec), min(testaccuracyvec))
    maxval = max(max(trainaccuracyvec), max(testaccuracyvec))
    plt.ylim(minval - (maxval - minval)/2.0, maxval + (maxval - minval)/2.0)
    plt.savefig('tcotp2_naivebayes_accuracy_'+ str(dictsize) +'.png',format='png')
    print trainaccuracyvec
    print testaccuracyvec

if __name__=='__main__':
  main()


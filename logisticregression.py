import numpy as np
import scipy.optimize as sc
import random as rand
import matplotlib.pyplot as plt
import math  

def main():
  dictsizes = [100, 1000, 5000]
  EPSILON = 1e-9
  NORMALIZEFEATURES = True
  
  for dictsize in dictsizes:
    traintcot = np.loadtxt('trainTCOT' + '_' + str(dictsize) + '.txt', dtype = int)
    trainp2 = np.loadtxt('trainP2' + '_' + str(dictsize) + '.txt', dtype = int)
    testtcot = np.loadtxt('testTCOT' + '_' + str(dictsize) + '.txt', dtype = int)
    testp2 = np.loadtxt('testP2' + '_' + str(dictsize) + '.txt', dtype = int)

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
    Ytrain[numtraintcot:] = -1
    
    Xtest = np.zeros( (numtesttcot + numtestp2, numfeatures) )
    Xtest[:numtesttcot,] = testtcot
    Xtest[numtesttcot:,] = testp2
    Ytest = np.zeros(numtesttcot + numtestp2)
    Ytest[:numtesttcot] = 1
    Ytest[numtesttcot:] = -1
    
    if NORMALIZEFEATURES:
      normXtrain = Xtrain.copy()
      normXtrainmean = np.mean(normXtrain, axis=0)
      normXtrainstd = np.std(normXtrain, axis = 0)
      normXtrain = (normXtrain - normXtrainmean ) / (EPSILON + normXtrainstd)
      
      normXtest = Xtest.copy()
      normXtestmean = np.mean(normXtest, axis=0)
      normXteststd = np.std(normXtest, axis = 0)
      normXtest = (normXtest - normXtestmean ) / (EPSILON + normXteststd)

    regularizationvec = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]#, 20, 50, 100, 1e7]
    trainaccuracyvec = []
    normtrainaccuracyvec = []
    testaccuracyvec = []
    normtestaccuracyvec = []
    for regularization in regularizationvec:
      print regularization
    
      LUCKYNUMBER = 40
      def sigmoid(x):
        if x < -1*LUCKYNUMBER: return 0
        if x > LUCKYNUMBER: return 1
        return 1.0/(1.0 + np.exp(-x))
        
      def logsigmoid(x):
        if x < -1*LUCKYNUMBER: return x
        if x > LUCKYNUMBER: return 0
        return -1*math.log(1.0 + np.exp(-x))
        
      def objectivefunction1(theta): #we minimize the objective function
        ret = 0.0
        for i in range(numtraintcot + numtrainp2): ret -= logsigmoid(Ytrain[i] * np.dot(Xtrain[i,:], theta))
        for i in range(1, numfeatures): ret += float(regularization) * (theta**2)
        return ret  
        
      def objectivefunction2(theta): #we minimize the objective function
        ret = 0.0
        for i in range(numtraintcot + numtrainp2): ret -= logsigmoid(Ytrain[i] * np.dot(normXtrain[i,:], theta))
        for i in range(1, numfeatures): ret += float(regularization) * (theta**2)
        return ret        

      def gradientfunction1(theta): #derivative of the objective function
        retvec = np.zeros(numfeatures)
        retvec[1:] = 2*regularization*theta[1:]
        for i in range(numtraintcot + numtrainp2): retvec = retvec - sigmoid(-1*Ytrain[i]*np.dot(Xtrain[i,:], theta))*Ytrain[i]*Xtrain[i,:]
        return retvec

      def gradientfunction2(theta): #derivative of the objective function
        retvec = np.zeros(numfeatures)
        retvec[1:] = 2*regularization*theta[1:]
        for i in range(numtraintcot + numtrainp2): retvec = retvec - sigmoid(-1*Ytrain[i]*np.dot(normXtrain[i,:], theta))*Ytrain[i]*normXtrain[i,:]
        return retvec
        
      theta1 = np.random.uniform(size=numfeatures)
      theta1, tmpminval, tmpd = sc.fmin_l_bfgs_b(objectivefunction1, theta1, fprime=gradientfunction1)

      if NORMALIZEFEATURES:
        theta2 = np.random.uniform(size=numfeatures)
        theta2, tmpminval, tmpd = sc.fmin_l_bfgs_b(objectivefunction2, theta2, fprime=gradientfunction2)
      
      #Get accuracy rate on both training and test sets
      trainaccuracy = 0.0
      normtrainaccuracy = 0.0
      testaccuracy = 0.0
      normtestaccuracy = 0.0
      for i in range(numtraintcot + numtrainp2):
        yhat = -1
        if np.dot(theta1, Xtrain[i,:]) > 0: yhat = 1
        if yhat == Ytrain[i]: trainaccuracy = trainaccuracy + 1.0
        
        if NORMALIZEFEATURES:
          normyhat = -1
          if np.dot(theta2, normXtrain[i,:]) > 0: normyhat = 1
          if normyhat == Ytrain[i]: normtrainaccuracy = normtrainaccuracy + 1.0
      trainaccuracyvec.append(trainaccuracy / float(numtrainp2 + numtraintcot))        
      if NORMALIZEFEATURES: normtrainaccuracyvec.append(normtrainaccuracy / float(numtrainp2 + numtraintcot))        
        
      for i in range(numtesttcot + numtestp2):
        yhat = -1
        if np.dot(theta1, Xtest[i,:]) > 0: yhat = 1
        if yhat == Ytest[i]: testaccuracy = testaccuracy + 1.0
        
        if NORMALIZEFEATURES:
          normyhat = -1
          if np.dot(theta2, normXtest[i,:]) > 0: normyhat = 1        
          if normyhat == Ytest[i]: normtestaccuracy = normtestaccuracy + 1.0        
      testaccuracyvec.append(testaccuracy / float(numtestp2 + numtesttcot))        
      if NORMALIZEFEATURES: normtestaccuracyvec.append(normtestaccuracy / float(numtestp2 + numtesttcot))        
      
    plt.clf()
    plt.plot(regularizationvec, trainaccuracyvec)
    plt.plot(regularizationvec, testaccuracyvec)
    legendvec = ['logreg-train', 'logreg-test']
    if NORMALIZEFEATURES:
      plt.plot(regularizationvec, normtrainaccuracyvec)
      plt.plot(regularizationvec, normtestaccuracyvec)
      legendvec.extend(['normalizedfeature-logreg-train', 'normalizedfeature-logreg-test'])
    plt.legend(legendvec, loc=4)
    plt.title('logistic regression accuracy on tcot/p2 for vocabulary size:'+str(dictsize))
    plt.ylabel('accuracy rate')
    plt.xlabel('regularization term')
    minval = min(min(trainaccuracyvec), min(testaccuracyvec))
    maxval = max(max(trainaccuracyvec), max(testaccuracyvec))
    if NORMALIZEFEATURES:
      minval = min(minval, min(min(normtrainaccuracyvec), min(normtestaccuracyvec)))
      maxval = max(maxval, max(max(normtrainaccuracyvec), max(normtestaccuracyvec)))
    plt.ylim(minval - (maxval - minval)/2.0, maxval + (maxval - minval)/2.0)
    plt.savefig('tcotp2_logreg_accuracy_'+ str(dictsize) +'.png',format='png')
    print trainaccuracyvec
    print testaccuracyvec
    if NORMALIZEFEATURES:
      print normtrainaccuracyvec
      print normtestaccuracyvec

if __name__=='__main__':
  main()



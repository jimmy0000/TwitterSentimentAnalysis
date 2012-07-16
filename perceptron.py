import numpy as np
import random as rand
import matplotlib.pyplot as plt

def main():
  dictsizes = [100, 1000, 5000]
  EPSILON = 1e-9
  
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
    Xtest = np.zeros( (numtesttcot + numtestp2, numfeatures) )
    Xtest[:numtesttcot,] = testtcot
    Xtest[numtesttcot:,] = testp2
    Ytest = np.zeros(numtesttcot + numtestp2)
    Ytest[:numtesttcot] = 1  
    
    MAXITERATIONS = 10000
    ITERATIONSFORPRINT = 250 
    
    w = np.random.uniform(size=2*numfeatures)
    wstore = np.zeros((MAXITERATIONS, 2*numfeatures))
    runningsumw = np.zeros(2*numfeatures)
    votedw = np.zeros(2*numfeatures)
    prevw = w.copy()
    Xtcot = np.zeros(2*numfeatures)
    Xp2 = np.zeros(2*numfeatures)
    
    perm = range(numtraintcot + numtrainp2)
    rand.seed()
    rand.shuffle(perm)
    permindex = 0
    
    iterationvec = []
    perceptrontrainaccuracyvec = []
    perceptrontestaccuracyvec = []
    votedperceptrontrainaccuracyvec = []
    votedperceptrontestaccuracyvec = []
    
    for iteration in range(MAXITERATIONS):
      wstore[iteration, :] = w.copy()
      runningsumw = runningsumw + wstore[iteration, :]
      
      if iteration % ITERATIONSFORPRINT == 0:
        #print 'iteration:%d'%(iteration)
        iterationvec.append(iteration)
        votedw = runningsumw / (iteration + 1)
        
        #Get accuracy rate on both training and test sets
        perceptrontrainaccuracy = 0.0
        votedperceptrontrainaccuracy = 0.0
        for i in range(numtraintcot + numtrainp2):
          Xtcot[:numfeatures] = Xtrain[i]
          yhattcot = np.dot(w, Xtcot)
          votedyhattcot = np.dot(votedw, Xtcot)
          Xp2[numfeatures:] = Xtrain[i]
          yhatp2 = np.dot(w, Xp2)
          votedyhatp2 = np.dot(votedw, Xp2)
          yhat = 1
          votedyhat = 1
          if yhatp2 > yhattcot + yhattcot * EPSILON: yhat = 0
          if votedyhatp2 > votedyhattcot + votedyhattcot * EPSILON: votedyhat = 0
          if yhat == Ytrain[i]: perceptrontrainaccuracy = perceptrontrainaccuracy + 1.0
          if votedyhat == Ytrain[i]: votedperceptrontrainaccuracy = votedperceptrontrainaccuracy + 1.0
        perceptrontrainaccuracyvec.append(perceptrontrainaccuracy / float(numtrainp2 + numtraintcot))        
        votedperceptrontrainaccuracyvec.append(votedperceptrontrainaccuracy / float(numtrainp2 + numtraintcot))        
        
        perceptrontestaccuracy = 0.0
        votedperceptrontestaccuracy = 0.0
        for i in range(numtesttcot + numtestp2):
          Xtcot[:numfeatures] = Xtest[i]
          yhattcot = np.dot(w, Xtcot)
          votedyhattcot = np.dot(votedw, Xtcot)
          Xp2[numfeatures:] = Xtest[i]
          yhatp2 = np.dot(w, Xp2)
          votedyhatp2 = np.dot(votedw, Xp2)
          yhat = 1
          votedyhat = 1
          if yhatp2 > yhattcot + yhattcot * EPSILON: yhat = 0
          if votedyhatp2 > votedyhattcot + votedyhattcot * EPSILON: votedyhat = 0
          if yhat == Ytest[i]: perceptrontestaccuracy = perceptrontestaccuracy + 1.0
          if votedyhat == Ytest[i]: votedperceptrontestaccuracy = votedperceptrontestaccuracy + 1.0
        perceptrontestaccuracyvec.append(perceptrontestaccuracy / float(numtestp2 + numtesttcot))        
        votedperceptrontestaccuracyvec.append(votedperceptrontestaccuracy / float(numtestp2 + numtesttcot))        
        
      
      Xtcot[:numfeatures] = Xtrain[perm[permindex]]
      yhattcot = np.dot(w, Xtcot)
      
      Xp2[numfeatures:] = Xtrain[perm[permindex]]
      yhatp2 = np.dot(w, Xp2)
      
      yhat = 1
      if yhatp2 > yhat + yhat * EPSILON: yhat = 0
      
      if yhat != Ytrain[perm[permindex]]:
        prevw = w.copy()
        w = prevw + (Ytrain[perm[permindex]] - yhat) * (Xtcot - Xp2)
      
      permindex = (permindex + 1) % (numtraintcot + numtrainp2)
      
    plt.clf()
    plt.plot(iterationvec, perceptrontrainaccuracyvec)
    plt.plot(iterationvec, perceptrontestaccuracyvec)
    plt.plot(iterationvec, votedperceptrontrainaccuracyvec)
    plt.plot(iterationvec, votedperceptrontestaccuracyvec)
    plt.legend(['perceptron-train', 'perceptron-test', 'voted-perceptron-train', 'voted-perceptron-test'], loc=4)
    plt.title('perceptron accuracy on tcot/p2 for vocabulary size:'+str(dictsize))
    plt.ylabel('accuracy rate')
    plt.xlabel('number of iterations')
    plt.ylim(0, max(max(max(perceptrontrainaccuracyvec), max(perceptrontestaccuracyvec)), max(max(votedperceptrontrainaccuracyvec), max(votedperceptrontestaccuracyvec))) + 0.1)
    plt.savefig('tcotp2_perceptron_accuracy_'+ str(dictsize) +'.png',format='png')
    print perceptrontrainaccuracyvec
    print perceptrontestaccuracyvec
    print votedperceptrontrainaccuracyvec
    print votedperceptrontestaccuracyvec

if __name__=='__main__':
  main()


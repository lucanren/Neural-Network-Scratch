import sys
import numpy as np
import csv
import pickle

filename = "mnist_train.csv"

inX = []
inY = []
with open(filename,'r') as file:
    reader = csv.reader(file)
    for row in reader:
        zeroes = [0 for x in range(0,10)]
        zeroes[int(row[0])] = 1
        inY.append(np.array([zeroes]))
        temp = row[1:]
        temp = [int(x)/255 for x in temp]
        inX.append(np.array([temp]))

def randomNet(lSizes): #lsizes is array corr to each layer [l0,l1,l2,l3]
    weights = [0]
    biases = [0]
    for i in range(1,len(lSizes)):
        currSize = lSizes[i]
        prevSize = lSizes[i-1]
        weights.append(2 * np.random.rand(prevSize, currSize) - 1)
        biases.append(2 * np.random.rand(1, currSize) - 1)
    return(weights,biases)

def sig(num):
    return 1/(1+np.exp(-1*num))
def dSig(num):
    return sig(num) * (1-sig(num))
def p_net(A,x,wList,bList):
    v_step = np.vectorize(A)
    aL = x
    for l in range(1,len(wList)):
        aL = v_step(aL@wList[l]+bList[l])
    return aL
def train(oWeights,oBiases,oX,oY,lbIn):
    n_sig = np.vectorize(sig)
    n_dSig = np.vectorize(dSig)
    lb = lbIn
    w = oWeights
    b = oBiases
    a = [0 for x in range(len(oWeights))]        
    dot = [0 for x in range(len(oWeights))]
    delta = [0 for x in range(len(oWeights))]

    for i in range(0,len(oX)):
        trainX = oX[i]
        trainY = oY[i]
        
        a[0] = trainX
        for layer in range(1,len(oWeights)):
            dot[layer] = a[layer-1]@w[layer]+b[layer]
            a[layer] = n_sig(dot[layer])
        # print(w)
        # print(b)
        # print(0.5*np.linalg.norm(trainY-a[-1])**2)
        # print()

        delta[-1] = n_dSig(dot[-1])*(trainY-a[-1])
        for layer in range(len(oWeights)-2,0,-1):
            delta[layer] = n_dSig(dot[layer])*(delta[layer+1]@(w[layer+1].transpose()))
        for layer in range(1,len(oWeights)):
            b[layer]=b[layer] + lb*delta[layer]
            #print(lb,(a[layer-1].transpose()),delta[layer])
            w[layer]=w[layer]+lb*(a[layer-1].transpose())@delta[layer]
        #print(str(i) + str(a[-1]))
        # print(w)
        # print(b)
        # te = p_net(sig,trainX,w,b)
        # print(0.5*np.linalg.norm(trainY-te)**2)
    return (w,b)

weights,biases = randomNet([784,300,100,10])



# with open('MNISTstoreW.pk', 'rb') as f:
#     weights = pickle.load(f)
# with open('MNISTstoreB.pk', 'rb') as f:
#     biases = pickle.load(f)

for j in range(1000):
        weights,biases = train(weights,biases,inX,inY,0.1)
        print("Epoch: " + str(j))
        print(p_net(sig,inX[0],weights,biases))
        print()
        with open("MNISTstoreW.pk", 'wb') as fi:
            pickle.dump(weights, fi)
        with open("MNISTstoreB.pk", 'wb') as fi:
            pickle.dump(biases, fi)

#print(biases)
miss = 0
for i in range(len(inX)):
    evalu = list(p_net(sig,inX[i],weights,biases)[0])
    ans = evalu.index(max(evalu))
    if ans != list(inY[i][0]).index(1): miss += 1
print(miss/len(inX))

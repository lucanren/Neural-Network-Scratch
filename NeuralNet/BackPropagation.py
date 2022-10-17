import sys
import numpy as np

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

def train(oWeights,oBiases,oX,oY,cNum,lbIn):
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
        if(cNum==2):
            print("Input vector at this step: " + str(trainX))
            print("Output vector at this step: " + str(a[-1]))
        # print(w)
        # print(b)
        # te = p_net(sig,trainX,w,b)
        # print(0.5*np.linalg.norm(trainY-te)**2)
    return (w,b)

def c1Check():
    weights = [0,np.array([[1,-0.5],[1,0.5]]),np.array([[1,2],[-1,-2]])]
    biases = [0,np.array([[1,-1]]),np.array([[-0.5,0.5]])]
    inX = [np.array([[2,3]])]
    inY = [np.array([[0.8,1]])]
    train(weights,biases,inX,inY)

def c2Sum():
    weights = [0,2 * np.random.rand(2, 2) - 1,2 * np.random.rand(2, 2) - 1]
    biases = [0,2 * np.random.rand(1, 2) - 1,2 * np.random.rand(1, 2) - 1]
    #print(weights)
    inX = [np.array([[0,0]]),np.array([[0,1]]),np.array([[1,0]]),np.array([[1,1]])]
    inY = [np.array([[0,0]]),np.array([[0,1]]),np.array([[0,1]]),np.array([[1,0]])]
    for x in range(5000):
        print("Epoch " + str(x) + ": ")
        weights,biases = train(weights,biases,inX,inY,2,0.2)

def c3Circle():
    weights = [0,2 * np.random.rand(2, 4) - 1,2 * np.random.rand(4, 1) - 1]
    biases = [0,2 * np.random.rand(1, 4) - 1,2 * np.random.rand(1, 1) - 1]
    #weights = [0,2 * np.random.rand(2, 2) - 1,2 * np.random.rand(2, 2) - 1]
    #biases = [0,2 * np.random.rand(1, 2) - 1,2 * np.random.rand(1, 2) - 1]
     
    inX = []
    inY = []
    with open("10000_pairs.txt") as f:
        for line in f:
            temp = line.split(" ")
            x1 = float(temp[0])
            x2 = float(temp[1])
            inX.append(np.array([[x1,x2]]))
            if(x1**2 + x2**2 < 1): inY.append(np.array([[1]]))
            else: inY.append(np.array([[0]]))
    for j in range(40):
        weights,biases = train(weights,biases,inX,inY,3,0.3)
        miss = 0
        for i in range(len(inX)):
            evalu = round(p_net(sig,inX[i],weights,biases)[0][0])
            if evalu != inY[i]: miss += 1
        print("Epoch " + str(j) + ": " + str(miss) + " missclassified")

whichC = sys.argv[1]
if whichC == "S":
    c2Sum()
if whichC == "C":
    c3Circle()

    
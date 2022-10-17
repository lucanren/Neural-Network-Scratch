import sys
import ast
import numpy as np


def step(num):
    if num>0:
        return 1
    return 0

def perceptron(A,w,b,x):
    out = 0
    for i in range(len(w)):
        out+=w[i] * x[i]
    out += b
    return A(out)

# def XOR(in1,in2): 
#     perceptron3 = perceptron(step,(-1,-2),3,(in1,in2))
#     perceptron4 = perceptron(step,(1,1),0,(in1,in2))
#     perceptron5 = perceptron(step,(1,1),-1.5,(perceptron3,perceptron4))
#     print(perceptron5)

def p_net(A,x,wList,bList):
    v_step = np.vectorize(A)
    aL = x
    for l in range(0,len(wList)):
        aL = v_step(aL@wList[l]+bList[l])
    return aL

#part 1
if(len(sys.argv)==2):
    inB = ast.literal_eval(sys.argv[1])
    i1 = inB[0]
    i2 = inB[1]
    inX = np.array([[i1,i2]])
    weights = [np.array([[-1,1],[-2,1]]),np.array([[1],[1]])]
    biases = [np.array([[3,0]]),np.array([[-1.5]])]
    print(p_net(step,inX,weights,biases)[0][0]) #XOR HAPPENS HERE

#part 2
if(len(sys.argv)==3):
    i1 = float(sys.argv[1])
    i2 = float(sys.argv[2])
    inX = np.array([[i1,i2]])
    weights2 = [np.array([[-1,-1,1,1,],[1,-1,-1,1]]),np.array([[1],[1],[1],[1]])]
    biases2 = [np.array([[1,1,1,1]]),np.array([[-3.5]])]
    print(p_net(step,inX,weights2,biases2)[0][0])

#part 3
if(len(sys.argv)==1):
    weights3 = [np.array([[-1,-1,1,1,],[1,-1,-1,1]]),np.array([[1],[1],[1],[1]])]
    biases3 = [np.array([[1.36,1.36,1.36,1.36]]),np.array([[-3.01]])]
    #print(p_net(step,inX,weights2,biases2))

    def sig(num):
        return 1/(1+np.exp(-1*num))

    coords = np.random.rand(500,2)*2-1
    #coords = [(-0.4388908087618326, 0.8991036131627772)]
    corr = 0
    wrong = []
    for row in coords:
        temp1 = p_net(sig,tuple(row),weights3,biases3)[0][0]
        temp = round(temp1)
        if temp == 1 and (row[0]**2 + row[1]**2)<1:
            corr+=1
        elif temp == 0 and (row[0]**2 + row[1]**2)>=1:
            corr += 1
        else:
            # print(temp1,temp,row[0]**2 + row[1]**2)
            # print(tuple(row))
            # print()
            wrong.append(tuple(row))
    print(wrong)
    print()
    print(corr/500)



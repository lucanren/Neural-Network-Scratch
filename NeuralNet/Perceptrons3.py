import sys
import ast

inB = ast.literal_eval(sys.argv[1])
i1 = inB[0]
i2 = inB[1]

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

def XOR(in1,in2): #XOR HAPPENS HERE
    perceptron3 = perceptron(step,(-1,-2),3,(in1,in2))
    perceptron4 = perceptron(step,(1,1),0,(in1,in2))
    perceptron5 = perceptron(step,(1,1),-1.5,(perceptron3,perceptron4))
    print(perceptron5)

XOR(i1,i2)
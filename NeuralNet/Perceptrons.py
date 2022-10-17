import sys
import ast
import matplotlib.pyplot as plt
import numpy as np
import math
import random

def truth_table(bits, n):
    binary = list(bin(n))[2:]
    while(len(binary) < 2**bits):
        binary.insert(0, 0)
    for x in range(len(binary)):
        binary[x] = int(binary[x])
    binary = tuple(binary)
    table = {}
    for x in range(2**bits-1,-1,-1):
        tableVal = list(bin(x).replace("b", "")[-bits:])
        for i in range(len(tableVal)):
            tableVal[i] = int(tableVal[i])
        while(len(tableVal) < bits):
            tableVal.insert(0, 0)
        tableVal = tuple(tableVal)
        table[tableVal] = binary[2**bits-x-1]
    return table

def pretty_print_tt(table):
    bits = len(list(table.keys())[0])
    for x in range(bits):
        print("In" + str(x), end = "   ")
    print("|  Out")
    for x in range(bits+1):
        print("______", end = "")
    print("")
    for x in table:
        for value in x:
            print(" " + str(value), end = "    ")
        print("|  " + str(table[x]))
    
def perceptron(A,w,b,x):
    num = 0
    for i in range(len(x)):
        num += x[i]*w[i]
    num += b
    return step(num)

def step(num):
    if(num > 0):
        return 1
    return 0

def stepD(num):
    if(num >= 0):
        return 1
    return 0
def sigmoid(num):
    return (1/(1+math.e**(-num)))
def check(n,w,b):
    table = truth_table(len(w), n)
    correct = 0
    total = 0
    for data in table:
        value = perceptron(step, w, b, data)
        if(value == table[data]):
            correct += 1
        total += 1
    return (correct/total)

def trainPerceptron(bits, n):
    w = [0 for x in range(bits)]
    b = 0
    prevEpoch = None
    epoch = {}
    for epochs in range(100):
        table = truth_table(bits,n)
        for value in table:
            obtainedVal = perceptron(step, w, b, value)
            difference = table[value] - obtainedVal
            for x in range(len(w)):
                w[x] += difference*value[x]
            b += difference
            epoch[value] = (w.copy(),b)
        if(epoch == prevEpoch):
            break
        prevEpoch = epoch
        epoch = {}
    return (w,b)


def perceptronModeling(bitsize):
    correctFunc = 0
    totalFunc = 2**(2**bitsize)
    for x in range(2**(2**bitsize)):
        w,b = trainPerceptron(bitsize, x)
        if(check(x,w,b) == 1):
            correctFunc += 1
    print(str(totalFunc) + " possible functions; " + str(correctFunc) + " can be correctly modeled.")
        

def p_net(A, x, wList, bList):
    vA = np.vectorize(A)
    a = [x]
    for i in range(len(wList)):
        a.append(vA(a[i] @ wList[i] + bList[i]))
    return a[-1]

def XOR(x):
    w1 = np.array([[2,-1], [2,-1]])
    b1 = np.array([-1,2])
    w2 = np.array([[1],[1]])
    b2 = np.array([-1])
    wList = [w1, w2]
    bList = [b1,b2]
    #XOR HAPPENS HERE
    return p_net(step, x, wList, bList)

def diamond(x):
    w1 = np.array([[1,1,-1,-1], [1,-1,-1,1],])
    b1 = np.array([-1,-1,-1,-1])
    w2 = np.array([[-1], [-1], [-1], [-1]])
    b2= np.array([0.5])
    wList = [w1, w2]
    bList = [b1, b2]
    return p_net(stepD, x, wList, bList)
    
def circle(x):
    w1 = np.array([[1,1,-1,-1], [1,-1,-1,1],])
    b1 = np.array([-1,-1,-1,-1])
    w2 = np.array([[-1], [-1], [-1], [-1]])
    b2= np.array([1.233])
    wList = [w1, w2]
    bList = [b1, b2]
    return round(p_net(sigmoid, x, wList, bList))

def challenge3():
    total = 500
    correct = 0
    misclassified = set()
    for x in range(500):
        x = random.random()*2-1
        y = random.random()*2-1
        coords = (x,y)
        if(x**2+y**2 < 1):
            if(circle(coords) == 1):
                correct += 1
            else:
                misclassified.add(coords)
        else:
            if(circle(coords) == 0):
                correct += 1
            else:
                misclassified.add(coords)
        
    for x in misclassified:
        print(x)
    print((correct/total))


#Part 1 Code
#number = int(sys.argv[1])
#x = ast.literal_eval(sys.argv[2])
#scalar = ast.literal_eval(sys.argv[3])
#print(check(number, x, scalar))

#Part 2 Code
#bitnum = int(sys.argv[1])
#number = int(sys.argv[2])
#w,b = trainPerceptron(bitnum,number)
#print("Weight vector: " + str(w))
#print("Bias Value: " + str(b))
#print("Accuracy: " + str(check(number, w, b)))

#Part 2 Graphing OW Code
#for n in range(16):
#    w,b = trainPerceptron(2,n)
#    xcoords = []
#    ycoords = []
#    for x in range(-20, 20):
#        for y in range(-20, 20):
#            i = (x/10,y/10)
#            if(perceptron(step,w,b,i) == 1):
#                plt.plot(x/10,y/10, "g.")
#            else:
#                plt.plot(x/10,y/10,"r.")
#    table = truth_table(2, n)
#    for a in table:
#        if(table[a] == 1):
#            plt.plot(a[0], a[1], "go")
#        else:
#            plt.plot(a[0], a[1], "ro")
#    plt.show()


#Part 4
commandline = sys.argv
if(len(sys.argv) == 2):
    x = ast.literal_eval(sys.argv[1])
    print(XOR(x))
elif(len(sys.argv) == 3):
    x = ast.literal_eval(sys.argv[1])
    y = ast.literal_eval(sys.argv[2])
    coords = (x,y)
    if(diamond(coords) == 1):
        print("inside")
    else:
        print("outside")
elif(len(sys.argv) == 1):
    challenge3()
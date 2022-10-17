import ast
import sys

def truth_table(bits,n):
    dic = dict()
    out = str(bin(n)[2:])
    while len(out)!=2**bits:
        out = "0" + out
    out = [int(x) for x in list(out)]

    bitList = [str(bin(x)[2:]) for x in range(2**bits)]
    bitList2 = []
    bitList3 = []
    for x in bitList:
        temp = str(x)
        while len(temp)!=bits:
            temp = "0" + temp
        bitList2.append(temp)
    bitList2.reverse()
    for b in bitList2:
        bitList3.append(tuple(int(x) for x in list(b)))
    for i in range(len(bitList3)):
        dic[bitList3[i]]=out[i]
    return dic

def pretty_print_tt(table):
    for x in table.keys():
        for i in range(len(x)):
            print(x[i],end=" ")
        print("| " + str(table[x]))

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

def check(n,w,b):
    tb = truth_table(len(w),n)
    cor = 0
    tot = 0
    for i in tb.keys():
        temp = perceptron(step,w,b,i)
        if temp == tb[i]: cor +=1
        tot +=1
    return cor/tot

inN = int(sys.argv[1])
inW = ast.literal_eval(sys.argv[2])
inB = float(sys.argv[3])

print(check(inN,inW,inB))
#pretty_print_tt(truth_table(2,5))
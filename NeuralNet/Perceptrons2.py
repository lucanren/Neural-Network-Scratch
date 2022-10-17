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

def check2(tb,w,b):
    cor = 0
    tot = 0
    for i in tb.keys():
        temp = perceptron(step,w,b,i)
        if temp == tb[i]: cor +=1
        tot +=1
    return cor/tot

def train(tb,targ):
    w = tuple([0 for x in range(len(list(tb.keys())[0]))])
    b = 0
    lamb = 1
    last = ()
    current = (w,b)
    count = 0
    while(last!=current):
        for i in tb.keys():
            fstar = perceptron(step,w,b,i)
            tempW = []
            #print(i)
            for j in range(len(i)):
                tempW.append(w[j]+(tb[i]-fstar)*lamb*i[j])
            w = tuple(tempW)
            #w = (w[0]+(tb[i]-fstar)*lamb*i[0],w[1]+(tb[i]-fstar)*lamb*i[1])
            b = b + (tb[i]-fstar) * lamb
            #print(current)
        last = current
        current = (w,b)
        count +=1
        if count == targ:
            return current
    return current
        
def possible(bits):
    tables = []
    out = []
    for i in range(0,2**(2**bits)):
        tables.append(truth_table(bits,i))
    for tb in tables:
        tempW,tempB = train(tb,100)
        out.append(check2(tb,tempW,tempB))
    #print(out)
    yes = len([1 for x in out if x == 1.0])
    poss = 2**(2**bits)
    print(str(poss) + " possible functions; " + str(yes) + " can be correctly modeled.")
            

    

#possible(4)

inBits = int(sys.argv[1])
inN = int(sys.argv[2])
w,b = train(truth_table(inBits,inN),100)
a = check(inBits,w,b)

print("1. The final weight vector: " + str(w))
print("2. The final bias value: " + str(b))
print("3. The accuracy of the perceptron as a decimal or percent: " + str(a))
# inN = int(sys.argv[1])
# inW = ast.literal_eval(sys.argv[2])
# inB = float(sys.argv[3])

# print(check(inN,inW,inB))
#pretty_print_tt(truth_table(2,5))
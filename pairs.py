import numpy as np
#import sklearn






def eval(test,data):
    err=0
    for a,num in data:
        err+=abs(np.count_nonzero((a-test)==0)-num)
    return err

def gen_option(data):
    x=data[-1][0]
    orig=x.copy()
    score=eval(x,data)
    i=0
    while score>0:
        i+=1
        if i%1000==0:
            x=orig
        tmp=x.copy()
        j,k=np.random.choice(10,2,False)
        t=tmp[j]
        tmp[j]=tmp[k]
        tmp[k]=t
        tmp_score=eval(tmp,data)
        if tmp_score<=score:
            score=tmp_score
            x=tmp

    return x,score

soln=np.arange(10)
np.random.shuffle(soln)

data=[]
x=np.arange(10)
num=np.count_nonzero((x-soln)==0)
data.append([x,num])
i=0
print(soln)
while sum(np.equal(x,soln))<10:
    i+=1
    x,_=gen_option(data)
    num=np.count_nonzero((x-soln)==0)
    data.append([x,num])
    print(x,num,i)
    

print(i)
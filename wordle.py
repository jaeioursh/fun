import numpy as np
import pickle
from itertools import product
from numba import jit,njit
#nltk.download()

def set_words(length=5):
    import nltk
    from nltk.corpus import words
    all=[]
    for w in words.words():
        if len(w)==length:
            all.append(w.lower())
    print(len(all))
    with open('data/words'+str(length)+'.pkl', 'wb') as outp:
        pickle.dump(all, outp, pickle.HIGHEST_PROTOCOL)
    #return all

def get_words(length):
    with open('data/words'+str(length)+'.pkl', 'rb') as inp:
        all = pickle.load(inp)
    return all

def word2mat(words):
    mat=[[ord(w) for w in word] for word in words]
    mat=np.array(mat)
    return mat
def word2vec(word):
    mat=[ord(w) for w in word] 
    mat=np.array(mat)
    return mat
def vec2word(vec):
    word=[chr(v) for v in vec]
    return "".join(word)

def inform(guess,answer):
    info=[]
    for i in range(len(guess)):
        if guess[i]==answer[i]:
            info.append(0)
        elif guess[i] in answer:
            info.append(1)
        else:
            info.append(2)
    return info
#@jit
def gen_mat(matrix):
    mat=np.zeros(( (len(matrix[0])+1,26,2,len(matrix) )),dtype=np.bool)
    
    for i in range(len(matrix[0])):
        for j in range(26):
            letter=j+ord("a")
            
            idx=matrix[:,i]==letter
            mat[i,j,0,:]= idx 

            idx1=matrix[:,i]==letter
            idx1=np.logical_not(idx1)
            idx2=np.any(matrix==letter,axis=1)
            idx=np.logical_and(idx1,idx2)
            mat[i,j,1,:]= idx 

            idx=np.any(matrix==letter,axis=1)
            idx=np.logical_not(idx)
            mat[-1,j,0,:]= idx 
    return mat

def reduce(guess,info,matrix):
    left=np.ones(matrix.shape[-1])
    for i in range(len(guess)):
        letter=guess[i]
        j=letter-ord("a")
        if info[i]==0:
            idx=matrix[i,j,0,:]
        if info[i]==1:
            idx=matrix[i,j,1,:]
        if info[i]==2:
            idx=matrix[-1,j,0,:]
        left=np.logical_and(left,idx)

    return left 


def information(matrix,word,left):
    if word is None:
        return np.log2(sum(left))
    
    vals=[]
    #print(infos)
    infos=[perm for perm in product(range(3),repeat=len(mat[0]))]
    infos=np.array(infos,dtype=np.int)
    for i in range(len(infos)):
        new_left=reduce(word,infos[i],matrix)
        vals.append(np.sum(np.logical_and(new_left,left)))
    
    vals=np.array(vals)
    vals=vals/np.sum(vals)
    vals=vals[vals!=0]
    bits=-vals*np.log2(vals)
    bits[np.isnan(bits)]=0
    return np.sum(bits)

def score(matrix,mat,left,mult=1):
    scores=[]
    i=0
    
    for m,l in zip(mat,left):
        if l:
            bits=information(matrix,m,left)
            i+=1
            scores.append([m,mult*bits])
            print(bits)
    scores=sorted(scores,key=lambda x:-x[1])
    for i in range(min(12,len(scores))):
        print(vec2word(scores[i][0]),scores[i][1])
    return scores[0][0]

def ai(matrix,mat,left,turn1=False):
    if turn1:
        guess=word2vec("raise") #tarie
        print(vec2word(guess))
    else:
        guess=score(matrix,mat,left)
    return guess

def cmd_line():
    x=input("score: ")
    return np.array([int(i) for i in x],dtype=int)

#set_words(5)
words=get_words(5)
mat=word2mat(words)
matrix=gen_mat(mat)
left=np.ones(len(mat))
print(matrix.shape)
#score(matrix,mat,left)
answer=mat[1000]

#print(information(mat,None,left))
#score(matrix,mat,left)
#for i in ["raise","store","adieu","crane","jazzy","qqqqq"]:
#    print(i,information(matrix,word2vec(i),left))
if 1:
    for i in range(6):
        guess=ai(matrix,mat,left,i==0)
        guess=word2vec(input("guess: "))
        print(vec2word(guess),information(matrix,guess,left))
        #info=inform(guess,answer)
        info=cmd_line()
        if sum(info)==0:
            break
        left*=reduce(guess,info,matrix)
        
        print(information(mat,None,left))

#guess=word2vec("hello")
'''
info=inform(guess,answer)
print(info)
print(len(mat))
print(information(mat))
infos=[perm for perm in product(range(3),repeat=len(mat[0]))]
infos=np.array(infos,dtype=np.int)
print(information(mat,guess,infos))
score(mat)
'''

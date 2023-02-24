import numpy as np
import sounddevice as sd

def gen_note(L,N):
    ffs=[15,22,15,18,20,7,2,5,7,9,4,2,3,4,5,6,5,5,7,5]
    ffs=np.array(ffs)/max(ffs)

    freq=440*2**(N/12)

    time=L/8
    sig=np.zeros(int(44100*time))
    for i in range(len(ffs)):
        i+=1
        t=np.linspace(0,time,int(44100*time))
        sig+=np.sin(np.pi*2*freq*i*t)

    sig/=len(ffs)*5
    sig*=np.exp(-t*10/L)
    return sig
sig=[]
#      a b c d e f g
minor=[0,2,3,5,7,8,10]
#         a b c d e f g#
harmonic=[0,2,3,5,7,8,11]
#      c d e f g  a  b
major=[3,5,7,8,10,12,14]
for i in range(100):
    L=np.random.choice(range(1,3))
    N=np.random.choice(harmonic)
    sig.append(gen_note(L,N-12))
'''
for i in range(4):
    for i in harmoic:
        sig.append(gen_note(1,i))
    for i in harmoic[::-1]:
        sig.append(gen_note(1,i))
'''
        
sig=np.hstack(sig)
print(sig)
sd.play(sig, 44100,blocking=True)
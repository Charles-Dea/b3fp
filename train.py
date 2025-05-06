import gymnasium as gym
import numpy as np
import os
import random
m='FFFFFFFFFFFFFFFFFFFHFFFFFFFFFHFFFFFHFFFFFHHFFFHFFHFFHFHFFFFHFFFG'
def Phi(s):
    match m[s]:
        case'G':
            return 1000000
        case'H':
            return-1000
        case'F':
            r=s//8
            c=s-r
            return 100-10*(8-r)-10*(8-c)
env=gym.make('FrozenLake-v1',map_name='8x8',is_slippery=True,render_mode='human')
env.metadata['render_fps']=2048
#generate a holy seed for better results
s=0
for c in 'INRI':s+=ord(c)
obs,inf=env.reset(seed=s)
if os.path.exists('q'):
    file=open('q','r')
    q=list(map(float,list(file)[-1].split()))
    file.close()
else:
    q=[0.0 for _ in range(256)]
file=open('q','w')
while True:
    for t in range(100000):
        a=random.choice(range(4))if random.random()>0.6 else np.argmax(q[obs*4:(obs+1)*4])
        no,r,trm,trnc,inf=env.step(a)
        q[obs*4+a]+=0.7*(r+0.9*Phi(no)-Phi(obs)+0.9*max(q[no*4:(no+1)*4])-q[obs*4+a])
        obs=no
        if obs==63:print('reached goal')
        if trm or trnc:obs,inf=env.reset()
    qs=''
    for i in q:qs+=str(i)+' '
    file.write(qs+'\n')
    

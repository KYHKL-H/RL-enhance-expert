import numpy as np

def reward_fun(data,data_new):
    C=np.sum(data[:,1,:],axis=1)
    C_new=np.sum(data_new[:,1,:],axis=1)
    D=np.sum(data[:,2,:],axis=1)
    D_new=np.sum(data_new[:,2,:],axis=1)

    reward=-((C_new-C)/(C+1)+(D_new-D)/(D+1))

    return reward
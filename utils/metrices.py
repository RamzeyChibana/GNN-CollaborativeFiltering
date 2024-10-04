import numpy as np




def Hit_at_k(batch_rates,k):
    batch_rates = batch_rates[:,:k]
    return np.sum(np.max(batch_rates,axis=1))


def Percision_at_k(batch_rates,k):
    batch_rates = batch_rates[:,:k]
    return np.sum(np.mean(batch_rates,axis=1))

def dgc_at_k(batch_rates,k):
    batch_rates = batch_rates[:,:k]
    log = np.log2(np.arange(1+1,k+2))
    
    return np.sum(batch_rates/log,axis=1)


def Ndgc_at_k(batch_rates,k):
    batch_rates = np.asfarray(batch_rates[:,:k])
    dgc = dgc_at_k(batch_rates,k)
    ideal_rates = np.sort(batch_rates,axis=1)[:,::-1]
    idgc = dgc_at_k(ideal_rates,k)
    ndgc = np.divide(dgc,idgc,out=np.zeros_like(dgc),where=idgc!=0)
    return np.sum(ndgc)






















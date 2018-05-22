
import numpy as np
import bigfloat
def calcula_mi(X):
    soma = 0
    for i in range(len(X)):
        soma += X[i]
    soma /= len(X)
    return soma

def calcula_sigma(X, mi):
    soma = 0
    n = len(X)

    for i in range(n):
        soma += X[i] * X[i].reshape(len(X[i]),1) - n * mi * mi.reshape(len(mi),1)
    soma /= n
    return np.asarray(soma,dtype=np.float64)

def calcula_posteriori(xk, mi, sigma):
    d = mi.shape[0]
    # print('xk',xk)
    # print('mi',mi)
    # print('sigma shape',sigma.shape,sigma)
    # print('sigma ', sigma)
    
    # inversa = np.linalg.inv(sigma)
    # determinante = 0.0
    # if(np.linalg.det(sigma) != 0.0):
    # sigma = np.array(sigma, dtype=np.float128)
    # print('np.linalg.det( np.linalg.inv(sigma) ) ', np.linalg.det( np.linalg.pinv(sigma) ))
    determinante = (2 * np.pi)**(-d/2) * np.linalg.det( np.linalg.pinv(sigma)) ** 1/2
    # print('determinante ' , determinante)
    #    determinante = np.power(np.linalg.det( np.linalg.inv(sigma)), 1/2)

    # determinante = 0.5
    xk = xk.reshape(1,len(xk))
    mi = mi.reshape(1,len(mi))

    diff = (xk - mi)
    parte3 = diff.reshape(1,diff.shape[1])
    # print('parte1.shape',parte3.shape)
    # print('parte1',parte3)

    parte1 = diff.reshape(diff.shape[1],1)
    # print('parte3.shape',parte1.shape)
    # print('parte3',parte1)

    parte2 = np.linalg.pinv(sigma)
    # print('parte2.shape',parte2.shape)

    mult1 = np.dot(parte2,parte1)
    # print('mult1.shape',mult1.shape)
    
    mult2 = np.dot(parte3, mult1)
    # print('mult2.shape',mult2.shape)

    exponencial = np.exp( (-1/2) * mult2 )

    # print('exponencial',exponencial)
    # print(1/0)
    return determinante * exponencial

# falta testar
def calcula_bayesianoGaussiano(xk, wName,prioridadesPriori,mis,sigmas):
    # print('wName',wName)
    # print('mis[wName]',mis[wName])
    # print('sigmas[wName]',sigmas[wName])
    # print('prioridadesPriori[wName]',prioridadesPriori[wName])
    posteriori = calcula_posteriori(xk,mis[wName],sigmas[wName])
    # print('posteriori ',posteriori)
    parte1 = posteriori * prioridadesPriori[wName]
    # print('parte1 - ', parte1)
    parte2 = 0
    for name in mis.keys():
        parte2 += calcula_posteriori(xk,mis[name],sigmas[name])
    
    return parte1 / parte2
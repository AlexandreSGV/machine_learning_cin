import numpy as np
import bigfloat

# X grupo em que se quer calcular a média
def calcula_mi(X):
    X = np.array(X)
    # print('X', X)
    soma = np.zeros(X.shape[1])


    for i in range(len(X)):
        # print('X[i]', X[i])
        soma += X[i]
    soma /= len(X)
    # print('mi', soma)

    return soma
# X é vetor com elementos do grupo (shape NxD)
# mi é média dos elementos do grupo
def calcula_sigma(X, mi):
    # print('X.shape', X.shape)
    X = np.array(X)
    soma = np.zeros((X.shape[1],X.shape[1]))
    n = len(X)

    for i in range(n):
        # soma += np.dot(X[i].reshape(1,len(X[i])) ,X[i].reshape(len(X[i]),1) ) - n * np.dot(mi.reshape(1,len(mi)),mi.reshape(len(mi),1))
        # soma += np.dot(X[i].reshape(len(X[i]),1),X[i].reshape(1,len(X[i]))  ) - n * np.dot(mi.reshape(len(mi),1) , mi.reshape(1,len(mi)) )

        diff = (X[i].reshape(len(X[i]),1) - mi.reshape(len(mi),1) )
        diffT = diff.reshape(1,diff.shape[0])
        # print('diff.shape', diff.shape)
        # print('diffT.shape', diffT.shape)
        # soma += np.dot( ( X[i].reshape(1, len(X[i])) - X[i].reshape(1,len(X[i])) ) ,  (mi.reshape(len(mi),1) - mi.reshape(1,len(mi)) ) )
        soma += np.dot(diff,diffT)
    soma /= n

    soma = soma * np.identity(soma.shape[1])

    # print(soma.diagonal())
    # este determinante da inversa não pode ser negativo, mas alguns det da inversa dão negativo.
    # dessa forma não se pode elevar sigma a 1/2 no cálculo da probabilidade a posteriori.
    # print('calcula_sigma.det da inv ', np.linalg.det(np.linalg.pinv(soma)))
    return soma



def calcula_posteriori(xk, mi, sigma):
    d = mi.shape[0]
    # print(d)
    # print('xk',xk)
    # print('mi',mi)
    # print('sigma shape',sigma.shape,sigma)
    # print('sigma ', sigma)

    # inversa = np.linalg.inv(sigma)
    # determinante = 0.0
    # if(np.linalg.det(sigma) != 0.0):
    # sigma = np.array(sigma, dtype=np.float128)
    # print('np.linalg.det( np.linalg.inv(sigma) ) ', np.linalg.det( np.linalg.pinv(sigma)))
    pi = ((2 * np.pi)**(-d/2))
    determinante =  (np.linalg.det( np.linalg.pinv(sigma)) ** 0.5)



    # determinante2 = ((2 * np.pi)**(-d/2)) * ( np.asarray(np.linalg.det( np.linalg.pinv(sigma)),dtype=np.float64) ** (0.5) )
    # print('determinante ' , determinante)

    # determinant =  (ownpow(np.linalg.det( np.linalg.pinv(sigma)) , 0.5))
    # print(' - - - - - - -')
    # print('np.linalg.det ' , np.linalg.det( np.linalg.pinv(sigma)))
    # print('type ' , type(np.linalg.det( np.linalg.pinv(sigma))))
    # print('determinant ' , determinant)
    #    determinante = np.power(np.linalg.det( np.linalg.inv(sigma)), 1/2)

    # determinant =  np.linalg.det( np.linalg.inv(sigma)) ** (0.5)
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
    return pi*determinante * exponencial

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
        parte2 += calcula_posteriori(xk,mis[name],sigmas[name]) *  prioridadesPriori[name]

    return parte1 / parte2

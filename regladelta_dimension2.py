# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 18:28:06 2026

@author: sofie
Este código implementa la regla delta en dimensión 2
"""

import matplotlib.pyplot as plt
import numpy as np

n=90 #número de puntos de entrenamiento
np.random.seed(65)
x=np.random.rand(2,n)*20-10
eta=0.1
epocas=150
x_test=np.random.rand(2,80)*20-10


def hardlim(x):
    return np.where(x >= 0, 1, 0)

#definir recta exacta
w_exact=[-1,1]
b_exact=2
s=w_exact@x+b_exact
s_test=w_exact@x_test+b_exact
#clasificar los datos de entrenamiento
Y=hardlim(s)
indices1=(Y==1)
indices2=(Y==0)
#clasificar datos de test
Y_test=hardlim(s_test)


#dibujar datos de entrenamiento
plt.scatter(x[0,indices1],x[1,indices1],color="red")
plt.scatter(x[0,indices2],x[1,indices2],color="blue")
plt.plot(x[0,],x[0,]-b_exact,color="pink",label="recta verdadera")
plt.grid(True, which='both', linestyle='-', alpha=0.7)

#entrenamiento
w=np.random.rand(2)
b=np.random.rand(1)
L_hist=[]

l_epocas=[]
for i in range(epocas):
   
    if i%10==0:
        z=w@x+b
        y_pred=hardlim(z)
        L=(1/n)*np.sum(np.abs(y_pred-Y))
        L_hist.append(L)
        l_epocas.append(i)
    
    if L==0:
        break
        
        
    for j in range(n):
        z=w@x[:,j]+b
        y_pred=hardlim(z)
        error=Y[j]-y_pred
        w=w+eta*error*x[:,j].T
        b=b+eta*error

z_test=w@x_test+b
y_pred_test=hardlim(z_test)
L_test=(1/80)*np.sum(np.abs(y_pred_test-Y_test))
print(f"error final {L_test}")
#dibujar recta aproximada
x0=np.min(x[0,:])
x1=np.max(x[0,:])
x_linea=np.array([x0,x1])
plt.plot(x_linea,-(w[0]*x_linea+b)/w[1],linestyle='--',linewidth=2,label="recta aproximada")
plt.legend()
plt.title("clasificación de puntos en dimensión 2")  
plt.xlabel(r"Eje $x_1$")
plt.ylabel(r"Eje $x_2$") 
plt.show()  

plt.plot(l_epocas,L_hist) 
plt.grid(True, which='both', linestyle='-', alpha=0.7) 
plt.xlabel(r"Número de iteraciones $i$")
plt.ylabel(r"Valor medio de error $E$")
plt.title("Variación de la función de error") 

        

    
    


# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 17:12:48 2026

@author: sofie
"""

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

#cargar datos
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# filtrar imagenes de 3 y 8
indices = (mnist_dataset.targets == 3) | (mnist_dataset.targets == 8) 
indice_test = (test_dataset.targets == 3) | (test_dataset.targets == 8)

img_data, label = mnist_dataset.data[indices], mnist_dataset.targets[indices]
img_test, label_test = test_dataset.data[indice_test], test_dataset.targets[indice_test]


# Normalizar a [0, 1] y convertir a vector
img_data = img_data.float() / 255.0 
img_flat = img_data.view(-1, 28*28).numpy()
num_3=0
#cambiar etiquetas a 0 y 1
label = label.numpy()
label_test=label_test.numpy()
label = np.where(label == 3, 1, 0)



    
epoca = 100
eta = 0.05
np.random.seed(42)
w = (np.random.rand(28*28) * 10) - 5
b = np.random.rand(1)
n = len(label)

def hardlim(x):
    return np.where(x >= 0, 1, 0)


hist_L=[]
l_epoca=[]
z = img_flat @ w + b
y_pred_total = hardlim(z)
L = (1/n) * np.sum(np.abs(y_pred_total - label))
print(f"Época {0} | Error (L): {L:.4f}")
hist_L.append(L)
l_epoca.append(0)

for i in range(1,epoca+1):
    
    #actualización de los pesos
    for j in range(n):
        z1 = w @ img_flat[j] + b 
        y_pred = hardlim(z1)
        error = label[j] - y_pred
        
        w = w + eta * error * img_flat[j]
        b = b + eta * error
        
    #imprimir progreso
    if i % 10 == 0:
        z = img_flat @ w + b
        y_pred_total = hardlim(z)
       
        L = (1/n) * np.sum(np.abs(y_pred_total - label))
        hist_L.append(L)
        l_epoca.append(i)
        print(f"Época {i} | Error (L): {L:.4f}")
        
    
plt.plot(l_epoca,hist_L,color="black",marker="o", linestyle="-")
plt.title("Variación de la función de error")   
plt.grid(True, which='both', linestyle='-', alpha=0.7)
plt.xlabel(r"Número de épocas $i$")
plt.ylabel(r"Valor medio de error $E$")
# datos de prueba
fig, axes = plt.subplots(1, 10, figsize=(15, 3))
  

for i in range(10):
    axes[i].imshow(img_test[i], cmap='gray')
    
    img_test1 = img_test[i].float() / 255.0
    img_test1 = img_test1.view(-1, 28*28).numpy()
    z_test = w @ img_test1.T + b
    y_pred_test = hardlim(z_test)

    prediccion_final = 3 if y_pred_test[0] == 1 else 8
   
    axes[i].set_title(f"pred:{prediccion_final}")
    axes[i].axis('off')
n1=len(label_test)
img_test = img_test.float() / 255.0 
img_test = img_test.view(-1, 28*28).numpy()
z_test = img_test @ w + b
y_pred_test = hardlim(z_test)
label_test = np.where(label_test == 3, 1, 0)

L_test = (1/n1) * np.sum(np.abs(y_pred_test - label_test))
print(f"error final {L_test}")
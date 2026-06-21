# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:03:21 2026

@author: sofie

Este código crea una figura para visualizar la demostración del Teorema  4.3.1.
"""


import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-1,1,400)
w = np.array([-0.8, -0.8]) #vector normal al hiperplano
a=0.85
y_sup=np.sqrt(1-x**2) #circunferencia
y_inf=-np.sqrt(1-x**2)

x_recta=np.linspace(-1.5,1.2,800) # puntos para graficar recta
y_recta_circulo=-(w[0]*x-a)/w[1]  #valor de la recta
x_hip=np.linspace(-1.5,1.2,800)

plt.figure(figsize=(7, 7))
plt.xlim(-2, 2)
plt.ylim(-2, 2)

limite_superior = np.maximum(y_inf, y_recta_circulo) #delimita la parte inferior de la zona rosa
plt.fill_between(x,  limite_superior, y_sup,
                 where=(y_sup > y_recta_circulo),#solo rellena entre limite_superior y y_sup donde se cumpla esa condicion
                 color="lightcoral", alpha=0.6, label="Zona activada")

limite_inferior = np.minimum(y_inf, y_recta_circulo) 
plt.fill_between(x, y_inf, limite_inferior, 
                 where=(y_inf < y_recta_circulo), #solo rellena si se cumple esa condicion
                 color="lightblue", alpha=0.6, label="Zona desactivada")


#dibujar semi circulo
norma_w=np.linalg.norm(w)
centro=(a/(norma_w**2))*w
r=np.sqrt(1-(a/norma_w)**2)
u=[-w[1],w[0]]/norma_w #eje del semicirculo
v=w/norma_w #eje del semicirculo
t_semi=np.linspace(0, np.pi, 200)
x_semi = centro[0] + r * np.cos(t_semi) * u[0] + r * np.sin(t_semi) * v[0] 
y_semi = centro[1] + r * np.cos(t_semi) * u[1] + r * np.sin(t_semi) * v[1]
plt.plot(x_semi, y_semi, color="blue", linewidth=3, label=r"Semicírculo de radio $r=\sqrt{1-\frac{a^2}{\|w\|^2}}$")
plt.fill(x_semi, y_semi, color="lightblue", alpha=0.5) 



t=np.linspace(0,1,100)
#dibujar hipotenusa
x_hipotenusa=(centro[0] + r * np.cos(0) * u[0])*t # (1-t)*(0,0)+t*(x_hipotenusa,y_hipotenusa)
y_hipotenusa=(centro[1] + r * np.cos(0) * u[1])*t
plt.plot(x_hipotenusa,y_hipotenusa,label="Hipotenusa")

#dibuajar uno de los catetos
x_cateto=((centro[0] + r * np.cos(0) * u[0]+centro[0] + r * np.cos(np.pi) * u[0])/2)*t
y_cateto=((centro[1] + r * np.cos(0) * u[1]+centro[1] + r * np.cos(np.pi) * u[1])/2)*t
plt.plot(x_cateto,y_cateto,label="Cateto 1")
plt.plot(x,y_sup,color="black")
plt.plot(x,y_inf,color="black")
plt.gca().set_aspect('equal')


x_recta=np.linspace(-1.5,1.2,800)
y_recta=-(w[0]*x-a)/w[1]
plt.plot(x_recta,-(w[0]*x_recta-a)/w[1],color="black",label=r"$w\cdot x = a$")
plt.plot(x,y_sup,color="black")
plt.plot(x,y_inf,color="black")
plt.arrow((centro[0] + r * np.cos(0) * u[0]+centro[0] + r * np.cos(np.pi) * u[0])/2, (centro[1] + r * np.cos(0) * u[1]+centro[1] + r * np.cos(np.pi) * u[1])/2, -0.3, -0.3, head_width=0.08, head_length=0.1, fc='red', ec='red', zorder=5)
plt.text(-0.75,- 0.85, r"$w$", fontsize=16, color='red', fontweight='bold')
plt.gca().set_aspect('equal')

plt.legend(loc="upper left")
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.title("Visualización 2D del Teorema 1", fontsize=14)



plt.show()
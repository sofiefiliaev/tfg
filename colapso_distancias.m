
%este código muestra como la distancia mínima y máxima es muy parecida en
%altas dimensiones

n=1000; %número de puntos

resultados=zeros(49,1);

for k=2:50 %k es la dimensión del espacio
    P=rand(n,k);
    origen=zeros(k,1);
    dist=zeros(n,1);
    for j=1:n
        dist(j)=norm(P(j,:)-origen);
    end

    
    dmin = min(dist(dist > 0.0001));
    dmax = max(dist(dist > 0.0001));
    resultados(k-1)=(dmax-dmin)/dmin;
end
% graficar los resultados
figure
plot(2:50,resultados,color="#000080",LineWidth=2)
title("Visualización del colapso de las distancias en alta dimensión","FontSize",18)  
xlabel("Dimensión","FontSize",16)
ylabel("$\frac{Dmax-Dmin}{Dmin}$","Interpreter","latex", "Fontsize", 16) 
grid on

    
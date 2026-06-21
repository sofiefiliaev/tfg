%este código muestra como en dimensiones mayores el volumen de una esfera
%se encuentra mayoritariamente cerca de la corteza

d=[1 2 4 6 20]; %dimensiones de los espacios
eps=linspace(0,1,100) ;
resultados=zeros(length(d),length(eps));

for i=1:length(d)
   
    for e=1:100
       
        frac=1-(1-eps(e))^d(i);
        resultados(i,e)=frac;
       
    end
end
%graficar los resultados para cada dimensión
figure;
hold on;
for i=1:length(d)
    dim = d(i);
    plot(eps, resultados(i, :), 'DisplayName', ['n= ' num2str(dim)], 'LineWidth', 2)
end

legend('Location', 'best','FontSize', 16);
title('Distribución del volumen de una esfera',"FontSize",18)  
xlabel('$\epsilon$', 'Interpreter', 'latex', 'FontSize', 16)
ylabel('$1-(1-\epsilon)^n$', 'Interpreter', 'latex', 'FontSize', 16) 
grid on;
hold off;
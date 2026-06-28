
%muestra una comparativa de separabilidad de puntos según la dimensión
m=1000; %número de puntos

eps=10^(-7);
relu = @(z) max(0, z);
n=50; %hasta que dimensión se realiza el experimento
ind_sep=zeros(n,1);
for d=1:n
    x=rand(m,d)*2-1;
    sep=0;
    for i=1:m
        theta=x(i,:)*x(i,:)'-eps;
        z=x*x(i,:)'-theta;
        c=relu(z); %vemos que puntos obtienen un producto escalar mayor que theta
        ind=sum(c>0);
        if ind==1
            sep=sep+1; %si solo el punto x_i obtiene z>0 entonces es separable
        end
    end
    ind_sep(d)=sep/m;
end
   
plot(1:n,ind_sep,LineWidth=2.5)
grid on
xlabel('Dimensión');
ylabel('Indice (Puntos separables/puntos totales) ');

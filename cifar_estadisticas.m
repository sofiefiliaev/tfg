%este código calcula estadísticas para una neurona selectiva
batch = load('cifar-10-batches-mat/data_batch_1.mat');
n=2000; %escoger 2000 imagenes
X=zeros(1024,n);
for i=1:n
    img=batch.data(i,:);

    R = reshape(img(1:1024),32,32)';
    G = reshape(img(1025:2048),32,32)';
    B = reshape(img(2049:3072),32,32)';

    Im = uint8(cat(3,R,G,B));

    aux = double(im2gray(Im));
    aux = aux(:) - mean(aux(:));
    X(:,i) = aux/max(abs(aux));
end
m=100; %repeticiones del experimento
num_exitos=zeros(m,1);
historial_inicial=zeros(m,1);
historial_final=zeros(m,1);
ratio=zeros(m,1);

tspan=[0,15];
%bucle para las repeticiones del experimento
for k=1:m
    wo=2*rand(1024,1)-1;
    prod=wo'*X;
    [~,ibest]=max(prod);
    b=(wo'*X(:,ibest))/2-1;
    
    alpha = 0.001;          
    relu = @(z) max(0, z - b);
    
    vo = relu(prod);
    historial_inicial(k)=sum(vo>0);
    
    [~,idx]=max(vo);

    s_p=X(:,idx);
    edo_neurona = @(t, w) alpha * relu(w'*s_p) * (w'*s_p) * (s_p - w * (w'*s_p));
   
    [t, w] = ode45(edo_neurona, tspan, wo);
    
    wend=w(end,:);
   
    
    v=relu(wend*X);
    indice_final=sum(v>0);
    historial_final(k)=indice_final;
    ratio(k)=indice_final/historial_inicial(k);
    if indice_final==1
        num_exitos(k)=1;
    end


end
fprintf("Estadísticas de las activaciones iniciales")
fprintf('  Media: %.2f | Desv. Estándar: %.2f\n', mean(historial_inicial), std(historial_inicial));
fprintf('  Mínimo: %d | Máximo: %d\n\n', min(historial_inicial), max(historial_inicial));

fprintf("Estadísticas de las activaciones finales")
fprintf('  Media: %.2f | Desv. Estándar: %.2f\n', mean(historial_final), std(historial_final));
fprintf('  Mínimo: %d | Máximo: %d\n\n', min(historial_final), max(historial_final));

%número de exitos
t=sum(num_exitos);
tasa=(t/m)*100;
fprintf("la tasa de exito es %d\n",tasa)
figure
plot(1:m,ratio,"o-") %muestra como varía el indice de activaciones según el experimento
grid on

xlabel('Número de experimento')
ylabel('Ratio de activaciones (Final / Inicial)')
%graficar el histogram
figure;
histogram(historial_final, 'BinMethod', 'integers');
xlabel('Número de imágenes activas al final');
ylabel('Frecuencia ');
title('Distribución de activaciones final');




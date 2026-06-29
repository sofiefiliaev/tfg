
%este codigo realiza el experimento de una neuronamselectivida usando
%estimulos principales distintos
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

wo=2*rand(1024,1)-1;
prod=wo'*X;
[~,ibest]=max(prod);
b=(wo'*X(:,ibest))/2-1;

alpha = 0.001;          
relu = @(z) max(0, z - b);

vo = relu(prod);
indice=sum(vo>0);
fprintf("inicialmente hay:%d\n",indice)
pos = find(vo > 0);
tspan=[0,15];
g=length(pos);
indice_dim=zeros(g,1);
indice_cop=zeros(g,1);
%probar con todos los estímulos que activan la neurona como principal
for i=1:g
    s_p=X(:,pos(i));
    edo_neurona = @(t, w) alpha * relu(w'*s_p) * (w'*s_p) * (s_p - w * (w'*s_p));
    [t, w] = ode45(edo_neurona, tspan, wo);
    wend=w(end,:);
    v=relu(wend*X);
    indice_final=sum(v>0);
    indice_dim(i)=indice_final/indice;
    indice_cop(i)=indice_final;
end
figure; % Crea una nueva ventana de figura
plot(1:g, indice_dim, '-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
xlabel('Estímulo elegido como principal')
ylabel('Ratio de activaciones (Final / Inicial)')
grid on;
figure
plot(1:g, indice_cop, '-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
grid on
xlabel('Estímulo elegido como principal')
ylabel('Número de activaciones finales')







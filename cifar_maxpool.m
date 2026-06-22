batch = load('cifar-10-batches-mat/data_batch_1.mat');
n=2000; % escoger 2000 imagenes

indice_cop=zeros(1, 4);
X=zeros(1024,n); 
X_16=zeros(256,n);
X_8=zeros(64,n);
X_4=zeros(16,n);

for i=1:n
    img=batch.data(i,:);
    R = reshape(img(1:1024),32,32)';
    G = reshape(img(1025:2048),32,32)';
    B = reshape(img(2049:3072),32,32)';
    Im = uint8(cat(3,R,G,B));
    aux = double(im2gray(Im)); % 'aux' inicialmente mide 32x32
   
    aux1 = aux(:) - mean(aux(:)); %aplanar la imagen y restarle la media
    X(:,i) = aux1/max(abs(aux1));  %queda normalizado entre -1 y 1
    
    
    aux_16=maxpool(aux, [2 2], 'Stride', 2); %16x16
    aux_vector16 = aux_16(:) - mean(aux_16(:));
    X_16(:,i) = aux_vector16/max(abs(aux_vector16)); 
    

    aux_8=maxpool(aux_16, [2 2], 'Stride', 2);
    aux_vector8 = aux_8(:) - mean(aux_8(:));
   
    X_8(:,i) = aux_vector8/max(abs(aux_vector8)); %dimension 8x8

    aux_4=maxpool(aux_8, [2 2], 'Stride', 2);
    aux_vector4 = aux_4(:) - mean(aux_4(:));
    X_4(:,i) = aux_vector4/max(abs(aux_vector4)); %dimension 4x4


    
end
%dimension 32x32
wo=2*rand(1024,1)-1;
prod=wo'*X;
[~,ibest]=max(prod);
b=(wo'*X(:,ibest))/2-1;

alpha = 0.001;          
relu = @(z) max(0, z - b);

vo = relu(prod);
indice=sum(vo>0);
fprintf("inicialmente hay:%d\n",indice)

[~,idx]=max(vo);

s_p=X(:,idx);
edo_neurona = @(t, w) alpha * relu(w'*s_p) * (w'*s_p) * (s_p - w * (w'*s_p));
tspan=[0,15];
[t, w] = ode45(edo_neurona, tspan, wo);

wend=w(end,:);
figure
plot(t,vecnorm(w,2,2))
grid on

v=relu(wend*X);
indice_final=sum(v>0);
fprintf("finalmente hay:%d\n",indice_final)
indice_cop(4)=indice_final/indice;

%dimension 16x16
wo=2*rand(256,1)-1;
prod=wo'*X_16;
[~,ibest]=max(prod);
b=(wo'*X_16(:,ibest))/2-1;

alpha = 0.001;          
relu = @(z) max(0, z - b);

vo = relu(prod);
indice=sum(vo>0);
fprintf("inicialmente hay:%d\n",indice)

[~,idx]=max(vo);

s_p=X_16(:,idx);
edo_neurona = @(t, w) alpha * relu(w'*s_p) * (w'*s_p) * (s_p - w * (w'*s_p));
tspan=[0,15];
[t, w] = ode45(edo_neurona, tspan, wo);

wend=w(end,:);
figure
plot(t,vecnorm(w,2,2))
grid on

v=relu(wend*X_16);
indice_final=sum(v>0);
fprintf("finalmente hay:%d\n",indice_final)
indice_cop(3)=indice_final/indice;



%dimension 8x8
wo = 2*rand(64,1) - 1;
prod=wo'*X_8;
[~,ibest]=max(prod);
b=(wo'*X_8(:,ibest))/2-1;

alpha = 0.001;          
relu = @(z) max(0, z - b);

vo = relu(prod);
indice=sum(vo>0);
fprintf("inicialmente hay:%d\n",indice)

[~,idx]=max(vo);

s_p=X_8(:,idx);
edo_neurona = @(t, w) alpha * relu(w'*s_p) * (w'*s_p) * (s_p - w * (w'*s_p));
tspan=[0,20];
[t, w] = ode45(edo_neurona, tspan, wo);

wend=w(end,:);
figure
plot(t,vecnorm(w,2,2))
grid on

v=relu(wend*X_8);
indice_final=sum(v>0);
fprintf("finalmente hay:%d\n",indice_final)
indice_cop(2)=indice_final/indice;


%dimension 4x4
wo = 2*rand(16,1) - 1;
prod=wo'*X_4;
[~,ibest]=max(prod);
b=(wo'*X_4(:,ibest))/2-1;

alpha = 0.001;          
relu = @(z) max(0, z - b);

vo = relu(prod);
indice=sum(vo>0);
fprintf("inicialmente hay:%d\n",indice)
% pos = find(vo > 0);
[~,idx]=max(vo);

s_p=X_4(:,idx);
edo_neurona = @(t, w) alpha * relu(w'*s_p) * (w'*s_p) * (s_p - w * (w'*s_p));
tspan=[0,25];
[t, w] = ode45(edo_neurona, tspan, wo);

wend=w(end,:);
figure
plot(t,vecnorm(w,2,2))
grid on

v=relu(wend*X_4);
indice_final=sum(v>0);
fprintf("finalmente hay:%d\n",indice_final)
indice_cop(1)=indice_final/indice;


%graficar resultados
figure
plot([16 64 256 1024],indice_cop, "-o",LineWidth=2)
grid on
xlabel('Dimensión (Número de píxeles)')
ylabel('Ratio de activaciones (Final / Inicial)')
title('Efecto del Max Pooling en la selectividad neuronal')


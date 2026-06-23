%este código construye una neurona selectiva usando la base de datos de
%cifar 
batch = load('cifar-10-batches-mat/data_batch_1.mat');
n=2000; %escoger 2000 imagenes
X=zeros(1024,n);
%convertir imagenes a vectores 
for i=1:n
    img=batch.data(i,:);

    R = reshape(img(1:1024),32,32)';
    G = reshape(img(1025:2048),32,32)';
    B = reshape(img(2049:3072),32,32)';
    %pasar a escala de grises
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
% pos = find(vo > 0);
[~,idx]=max(vo);

s_p=X(:,idx);
edo_neurona = @(t, w) alpha * relu(w'*s_p) * (w'*s_p) * (s_p - w * (w'*s_p));
tspan=[0,5];
[t, w] = ode45(edo_neurona, tspan, wo);

wend=w(end,:);
figure
plot(t,vecnorm(w,2,2),LineWidth=2,Color="black")
grid on
xlabel('Tiempo','Interpreter', 'latex');
ylabel('Norma de los pesos $w$', 'Interpreter', 'latex');

figure
%muestra el coseno del ángulo entre w y s_p
prod_esc=w*s_p./(vecnorm(w,2,2)*norm(s_p));
plot(t,prod_esc',LineWidth=2,Color="black")
xlabel('Tiempo','Interpreter', 'latex');
ylabel(' $ \frac{w(t) \cdot s_p}{\|w(t)\|\|s_p\|}$', 'Interpreter', 'latex');
grid on

%comprobar cuantas activaciones finales hay
v=relu(wend*X);
indice_final=sum(v>0);
fprintf("finalmente hay:%d\n",indice_final)







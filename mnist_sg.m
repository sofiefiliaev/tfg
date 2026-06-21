%este codigo implementa el descenso del gradiente estocástico

clear all
%procesar los datos
datos = readmatrix('mnist_train.csv');
datos_test = readmatrix('mnist_test.csv');
Xtest= datos_test(:,2:end);
Xtrain=datos(:,2:end);
Ytrain=datos(:,1);
Ytest=datos_test(:,1);
numImagenes = size(Xtrain, 1);
numtest=size(Xtest,1);


indices=(Ytrain==3) | (Ytrain==8);
indices_test=(Ytest==3) | (Ytest==8);

X=double(Xtrain(indices,:))/255;
Xtest=double(Xtest(indices_test,:))/255;

Ytrain_original=Ytrain(indices);
Ytest_original=Ytest(indices_test);

Y=double(Ytrain_original==3);
Ytest_final=double(Ytest_original==3);



%definir función sigmoide
sigmoide=@(x) 1 ./ (1 + exp(-x));


iteraciones = 5000;
eta = 0.01; % tasa de aprendizaje


w = rand(1, 28*28) * 0.1; % vector de pesos (1 x 784)
b = 0; % inicializar b
n = length(Y);
m = 1; % tamaño del lote


hist_L = zeros(1, iteraciones);
l_epoca = 1:iteraciones;
for i=1:iteraciones
    lote = randi([1, n], 1, m);
    
    X_batch = X(lote, :); % (m x 784)
    Y_batch = Y(lote);    % (m x 1)
    
    
    z = (X_batch * w') + b;     % (m x 1)
    y_pred = sigmoide(z); 
    
    % cálculo de error y derivadas
    error = y_pred - Y_batch; %(mx1)
    deriv_sigmoide = y_pred .* (1 - y_pred);
    
    L = mean(error.^2);
    
    % parciales 
    dw = (2/m) * ((error .* deriv_sigmoide)' * X_batch); % (1 x 784)
    db = (2/m) * sum(error .* deriv_sigmoide);
    
    if mod(i, 100) == 0
        fprintf('Iteraciones %d | Error (L): %.4f\n', i, L);
    end
    
    hist_L(i) = L; 
    % actualización
    w = w - eta * dw;
    b = b - eta * db;
end

figure;
plot(l_epoca, hist_L, 'k', 'LineWidth', 1.5);
title(sprintf('Variación de la función de error para m=%d', m),fontsize=18);
grid on;
xlabel('Número de iteraciones i');
ylabel('Error cuadrático medio E');

% comprobar precisión
z_test_all = (Xtest * w') + b;
y_pred_test_all = sigmoide(z_test_all);

prediccion_final_test = double(y_pred_test_all >= 0.5);
acc = mean(Ytest_final == prediccion_final_test);

fprintf('\nPrecisión: %.4f\n', acc);

L_test = mean((y_pred_test_all - Ytest_final).^2); %error cuadrático medio final
fprintf('Error final: %.4f\n', L_test);


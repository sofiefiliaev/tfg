%este código implementa el denseso del gradiente estocástico pero gráfica
%por épocas.

clear all
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

Y=double(Ytrain_original==3); %(n x 1)
Ytest_final=double(Ytest_original==3);

%definir función sigmoide
sigmoide = @(x) 1 ./ (1 + exp(-x));

epocas = 10;
eta = 0.1; % tasa de aprendizaje
w = rand(1, 28*28) * 0.1; % vector de pesos (1 x 784)
b = 0; % inicializar b
n = length(Y);
m = n; % tamaño del lote

iteraciones_por_epoca = ceil(n/m);


hist_L_epoca = zeros(1, epocas); 


for epoca = 1:epocas 
    indices=randperm(n);
    L_acumulado = 0;
    
    % bucle para las iteraciones
    for i = 1:m:n
        final=min(i+m-1,n); %escoger final del lote para que no se pase del indice n
        lote = indices(i:final);
        l=length(lote);
        
        
        X_batch = X(lote, :); % (m x 784)
        Y_batch = Y(lote);    % (m x 1)
        
       
        z = (X_batch * w') + b;     
        y_pred = sigmoide(z); %(m x 1)
        
        % cálculo de error y derivadas
        error = y_pred - Y_batch; 
        deriv_sigmoide = y_pred .* (1 - y_pred);
        
       
        L_acumulado = L_acumulado + sum(error.^2);
        
        % parciales 
        dw = (2/l) * ((error .* deriv_sigmoide)' * X_batch); 
        db = (2/l) * sum(error .* deriv_sigmoide);
        
        % actualización
        w = w - eta * dw;
        b = b - eta * db;
    end
   
    error_medio_epoca = L_acumulado / (iteraciones_por_epoca * m);
    hist_L_epoca(epoca) = error_medio_epoca; 
    
   
    fprintf('Época %d | Error Medio (L): %.4f\n', epoca, error_medio_epoca);
    
end

figure;

plot(1:epocas, hist_L_epoca, 'k-o', 'LineWidth', 1.5); 
title(sprintf('Variación del error por época (m=%d)', m), 'FontSize', 14);
grid on;
xlabel('Número de Épocas');
ylabel('Error cuadrático medio E');

% comprobar precisión
z_test_all = (Xtest * w') + b;
y_pred_test_all = sigmoide(z_test_all);
prediccion_final_test = double(y_pred_test_all >= 0.5);
acc = mean(Ytest_final == prediccion_final_test);
fprintf('\nPrecisión en Test: %.4f\n', acc);
%error cuadrático medio
L_test = mean((y_pred_test_all - Ytest_final).^2);
fprintf('Error final en Test: %.4f\n', L_test);
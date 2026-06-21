%este código muestra 10 imagenes del conjunto CIFAR-10
batch = load('cifar-10-batches-mat/data_batch_1.mat');
n=10; %escoger 2000 imagenes
X=zeros(32,32,3,n,"uint8");
for i=1:n
    img=batch.data(i,:);

    R = reshape(img(1:1024),32,32)';
    G = reshape(img(1025:2048),32,32)';
    B = reshape(img(2049:3072),32,32)';

    X(:, :, 1, i) = R; 
    X(:, :, 2, i) = G; 
    X(:, :, 3, i) = B; 
end

figure;
t = tiledlayout(2,5); 
t.TileSpacing = 'compact';

for i=1:n

    nexttile;
    img_color = X(:, :, :, i); 
    imshow(img_color);
    
end


clear;
clc
close all;
addpath(genpath('./dataset'));
In = imread("1.png");
[M,N,C] = size(In);
B = zeros(2,2,2);
T = zeros(M,N,C);
Ib = zeros(M,N,C);
Ir = zeros(M,N,C);
Y = zeros(M,N);
%F = [1 2;2 3];
F = [2 1;3 2];
B(:,:,1) = F==1;
B(:,:,2) = F==2;
B(:,:,3) = F==3;
I = ones(M/2,N/2);
for c=1:C
    T(:,:,c) = kron(I,B(:,:,c));
    Y = Y + T(:,:,c).*double(In(:,:,c));
end
kernel = zeros(3,3,3);
B = [1 2 1; 2 4 2;1 2 1];
R = [1 2 1; 2 4 2;1 2 1];
G = [0 1 0;1 4 1;0 1 0];
kernel(:,:,1) = B;
kernel(:,:,2) = G;
kernel(:,:,3) = R;
subplot(1,3,1),imagesc(Y)
subplot(1,3,2),imagesc(In)

for c = 1:C
    Ib(:,:,c) = T(:,:,c).*Y;
    Ir(:,:,c) = imfilter(Ib(:,:,c),kernel(:,:,c)/4,"symmetric");
end

% for c = 1:C
%     Ib(:,:,c) = T(:,:,c).*Y;
%     X =  Ib(:,:,c);
%     %Ir(:,:,c) = imfilter(Ib(:,:,c),kernel(:,:,c),);
%     [x,y,v] = find(X);
%     [xq,yq] = find(X==0);
%     F = scatteredInterpolant(x,y,v,'natural');
%     X(X==0) = F(xq,yq);
%     Ir(:,:,c) = X;
% end
Ir = uint8(Ir);
psnr(Ir,In)
ssim(Ir,In)
%sam(Ir,In)
subplot(1,3,3),imagesc(uint8(Ir))
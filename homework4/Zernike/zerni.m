function [Z] = zerni(N,nModes,delta)

[x, y] = meshgrid((-N/2 : N/2-1) * delta);
[t, r] = cart2pol(x, y);
Z = zeros(N*N, nModes);

for s=1:nModes
    temp = zernike2(s, r, t);
    Z(:,s) = temp(:);
end

end
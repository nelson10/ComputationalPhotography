function [W1] = zernikeLinearCombination(Z,A)
W = Z*A;
N = sqrt(length(Z));
W1 = reshape(W,[N,N]);
W1 = mat2gray(W1);
end



function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

ez=2.71828.^z;
ezreciproc=ez.^-1;
oneplusezminusone=1.+ezreciproc;
g=oneplusezminusone.^-1;
% =============================================================

end

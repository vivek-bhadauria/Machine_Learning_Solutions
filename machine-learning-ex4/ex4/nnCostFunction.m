function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X=[ones(m,1) X];
layer2Features=sigmoid(X*Theta1');
layer2Features=[ones(m,1) layer2Features];
hypothesis=sigmoid(layer2Features*Theta2');

y_matrix = eye(num_labels)(y,:);

J=(-1/m)*sum(sum(log(hypothesis).*y_matrix+log(1-hypothesis).*(1-y_matrix)));

Theta1square=Theta1.^2;
Theta2square=Theta2.^2;
regulztnFactor=(lambda/(2*m))*(sum(sum(Theta1square))-sum(Theta1square(:,1)) + sum(sum(Theta2square))-sum(Theta2square(:,1)));
J = J+regulztnFactor;

y_labels=1:num_labels;
delta1 = zeros(size(Theta1)); %25*401
delta2 = zeros(size(Theta2)); %10*26
for t=1:m,
    a_1=X(t,:); %1x401
    z_2=a_1*Theta1'; %1X25
    a_2=sigmoid(z_2); %1X25
    a_2=[1 a_2]; %1x26
    %size(a_2)
    z_3=a_2*Theta2'; %1X10
    %size(z_3)
    a_3=sigmoid(z_3); %1X10
    a_3=a_3'; %10x1
    y_vector=(y_labels==y(t))'; %10X1

    %size(y_vector)
    d3=a_3-y_vector; %delta calculated for the third layer and it is a 10x1 vector
    g_dash_z2=sigmoidGradient(z_2); %1X25
    g_dash_z2=[1 g_dash_z2]; %1x26
    d2=(Theta2'*d3).*g_dash_z2'; %Theta2'(26x10),d3(10x1) so Theta2'*d3(26x1).*g_dash_z2'(26x1) and result is d2 (26x1)

    delta2 = delta2 + (d3*a_2);
    delta1 = delta1 + (d2*a_1)(2:end,:);

end;

Theta1_grad=(1/m)*delta1;
Theta2_grad=(1/m)*delta2;

%Regularization of gradient params
%for delta1
for q=1:size(Theta1_grad,1),
    for r=2:size(Theta1_grad,2),
      Theta1_grad(q,r)=Theta1_grad(q,r)+(lambda/m)*Theta1(q,r);
    end; 
end;
%for delta2
for q=1:size(Theta2_grad,1),
    for r=2:size(Theta2_grad,2),
      Theta2_grad(q,r)=Theta2_grad(q,r)+(lambda/m)*Theta2(q,r);
    end; 
end;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

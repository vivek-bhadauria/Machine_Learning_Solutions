function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%









% =========================================================================
y1=find(y==1);
y0=find(y==0);
plot(X(y1,1),X(y1,2),"+","color","r","markersize",7,"linewidth",2);
plot(X(y0,1),X(y0,2),"*","color","y","markersize",7,"linewidth",2);


hold off;

end

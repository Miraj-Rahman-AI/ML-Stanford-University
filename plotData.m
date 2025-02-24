function plotData(x, y)
%PLOTDATA Plots the data points x and y into a new figure
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of 
%   population and profit.

figure; % open a new figure window

% Plot the data points
plot(x, y, 'rx', 'MarkerSize', 10);

% Set the axes labels
xlabel('Population of City in 10,000s');
ylabel('Profit in $10,000s');

% ============================================================

end
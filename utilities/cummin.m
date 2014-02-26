function x = cummin(x) 
% y = cummin(x)
%   finds the cummulative minimum of x
%   e.g. y_i = min( x_i, y_{i-1} )

if numel(x) > length(x)
    error('input must be a vector');
end
for k = 2:length(x)
    x(k) = min( x(k), x(k-1) );
end
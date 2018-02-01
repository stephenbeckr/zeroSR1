% Meant to be called by getReferenceSolution.m
% This file is NOT compatible with Octave

fprintf('Computing reference solution via CVX\n');
cvx_precision best
cvx_quiet true

switch problemName
    case 'simple_001'
        cvx_begin
        variable xRef(N)
        minimize sum_square(A*xRef-b)/2 + lambda*norm(xRef,1)
        cvx_end
    case 'simple_002' % same setting, different parameters
        cvx_begin
        variable xRef(N)
        minimize sum_square(A*xRef-b)/2 + lambda*norm(xRef,1)
        cvx_end
end
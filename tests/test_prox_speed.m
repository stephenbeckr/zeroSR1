%{
Test the speed of the various projections, as a function of input size
The paper claims it is O( n log n), so we verify that here.

We test 5 proxes, and also compare to the time it takes to sort n numbers,
and also compare to O(n) and O(n log n) lines.

The results: the scaled prox algorithms take about 10x the time
to sort n numbers. Not bad.

Stephen Becker, Feb 26 2014 stephen.beckr@gmail.com
%}
nReps   = 5;
nList   = logspace(2,7,6);
typeList = {'l1','Rplus','box','hinge','linf','sort'};  nTypes = length(typeList);
RESULTS = zeros(nTypes,length(nList),nReps);
INVERT  = true; % must be true for now...
for ni = 1:length(nList)
    n = nList(ni);
    fprintf('Test %d of %d: n = %d\n', ni, length(nList), n );
    for ri = 1:nReps
        d = rand(n,1);
        u = 10*randn(n,1);
        y = randn(n,1);
        offset  = randn(n,1);
        lambda  = 9;
        lwr     = randn(n,1);
        upr     = lwr + 2*rand(n,1);
        
        for type_i = 1:nTypes
            type = typeList{type_i};
            
            switch lower(type)
                case 'l1'
                    prox            = @(x,t) sign(x).*max(0, abs(x) - t );
                    prox_brk_pts    = @(s) [-s,s];
                case 'rplus'
                    prox            = @(x,t) max(0, x);
                    prox_brk_pts    = @(s) 0; % since projection, scaling has no effect
                case 'box'
                    prox            = @(x,t) max( min(upr,x), lwr );
                    prox_brk_pts    = @(s) [lwr,upr]; % since projection, scaling has no effect
                case 'hinge'
                    prox    = @(x,t) 1 + (x-1).*( x > 1 ) + (x + t - 1).*( x + t < 1  );
                    prox_brk_pts    = @(s)[ones(size(s)), 1-s];
                case 'linf'
                    prox            = @(x,t) sign(x).*min( 1, abs(x) );
                    prox_brk_pts    = @(s) [-ones(size(s)),ones(size(s))];
                case 'sort'
            end
            if strcmpi(type,'sort') % a baseline measure of speed
                t2 = tic;
                x = sort( y );
                tm2 = toc(t2);
            else
                scaledProx = @(varargin) prox_rank1_generic( prox, prox_brk_pts, varargin{:}); 
                t2 = tic;
                x = scaledProx( y, d, u, lambda, offset, 1, INVERT);
                tm2 = toc(t2);
            end
            RESULTS( type_i, ni, ri ) = tm2;
        end
    end
end
%% Plot
figure(1); clf;
times = median(RESULTS,3);
h=loglog( nList, times', 'o-' );
set(h(end),'marker','*')
xlabel('Dimension "n" of input');
ylabel('Time to solve, in seconds');
% Add a line of n
hold all
ref = 3; % which point to reference
loglog( nList, nList*times(1,ref)/nList(ref), '--','linewidth',2 );
loglog( nList, nList.*log2(nList)*times(1,ref)/(nList(ref).*log2(nList(ref))), '--','linewidth',2 );
loglog( nList, nList.^2*times(1,ref)/(nList(ref)^2), '--','linewidth',2 );
legend( {typeList{:}, 'O(n)','O(n log n)','O(n^2)'}, 'location','northwest' )
ylim([1e-3,20]);
title('Time to compute the scaled prox, median of 5 runs');
%% Save as a file
% set(gcf, 'PaperPositionMode', 'auto');
% print -dpng test_prox_speed.png
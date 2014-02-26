function emphasizeRecent
% Makes the most recent line-series in bold, and all the others
% are not in bold. Call this with no arguments or outputs.
% Written by Stephen Becker, stephen.beckr@gmail.com  2011

list = get(gca,'children');

% Make everything else normal width
% i.e. undo any previous calls of emphasizeRecent()
set( list, 'linewidth', 0.5 );

% Make most recent item in bold
set( list(1), 'linewidth',2);

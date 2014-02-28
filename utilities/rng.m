function varargout = rng(varargin)
%RNG Control the random number generator used by RAND, RANDI, and RANDN (modified version)
%   Newer versions of matlab use 'rng' to set the seed of random number generators.
%   This makes code that calls 'rng' not backward compatible, so this function
%   is provided instead.
%   If 'rng' exists on your system, then that will be called, otherwise
%   it defaults to the old style of setting random number seeds
%   (and unfortunately I cannot guarantee that the two versions will set
%    equivalent seeds, but at least, on a single computer, your runs will always
%   b e consistent).

%   See <a href="matlab:helpview([docroot '\techdoc\math\math.map'],'update_random_number_generator')">Updating Your Random Number Generator Syntax</a> to use RNG to replace
%   RAND or RANDN with the 'seed', 'state', or 'twister' inputs.


%   See <a href="http://www.mathworks.com/access/helpdesk/help/techdoc/math/brn4ixh.html#brvku_2">Choosing a Random Number Generator</a> for details on these generators.

persistent do_once
% In 2014, rng is not builtin, it's in a package, so be careful.
%   so unfortunately cannot call "builtin('rng')"
C = which('rng','-all');
if size(C,1) > 1 && isempty(do_once)
    do_once = true;
    % add this directory to the very top of the path so it shadows this
    % file...
    addpath(fileparts( C{2} ) )
    [varargout{1:nargout}] = rng( varargin{:} );
    return;
end

if exist('rng','builtin')
    [varargout{1:nargout}] = builtin('rng',varargin{:} );
    return;
end

% Otherwise, we have an old version of Matlab...
error(nargchk(1,1,nargin));
error(nargoutchk(0,0,nargout));
arg1 = varargin{1};
% For R2008a, this doesn't work... (not sure what earliest version is)
if verLessThan('matlab','7.7')
    % VERY old matlab
    randn('state',arg1);
    rand('state',arg1);
else
    % Somewhat old matlab
    RandStream.setDefaultStream(RandStream('mt19937ar', 'seed', arg1 ));
end

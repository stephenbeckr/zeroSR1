function varargout = rng(varargin)
%RNG Control the random number generator used by RAND, RANDI, and RANDN (SRB version)
%   RNG(SD) seeds the random number generator using the non-negative
%   integer SD so that RAND, RANDI, and RANDN produce a predictable
%   sequence of numbers.
%
%   RNG('shuffle') seeds the random number generator based on the current
%   time so that RAND, RANDI, and RANDN produce a different sequence of
%   numbers after each time you call RNG.
%
%   RNG(SD,GENERATOR) and RNG('shuffle',GENERATOR) additionally specify the
%   type of the random number generator used by RAND, RANDI, and RANDN.
%   GENERATOR is one of:
%
%       Generator              Description
%      ------------------------------------------------------------------
%      'twister'               Mersenne Twister
%      'combRecursive'         Combined Multiple Recursive
%      'multFibonacci'         Multiplicative Lagged Fibonacci
%      'v5uniform'             Legacy MATLAB 5.0 uniform generator
%      'v5normal'              Legacy MATLAB 5.0 normal generator
%      'v4'                    Legacy MATLAB 4.0 generator
%
%   RNG('default') puts the settings of the random number generator used by
%   RAND, RANDI, and RANDN to their default values so that they produce the
%   same random numbers as if you restarted MATLAB. In this release, the
%   default settings are the Mersenne Twister with seed 0.
%
%   S = RNG returns the current settings of the random number generator
%   used by RAND, RANDI, and RANDN. The settings are returned in a
%   structure S with fields 'Type', 'Seed', and 'State'.
%    
%   RNG(S) restores the settings of the random number generator used by
%   RAND, RANDI, and RANDN back to the values captured previously by
%   S = RNG.
%
%   S = RNG(...) additionally returns the previous settings of the random
%   number generator used by RAND, RANDI, and RANDN before changing the
%   seed, generator type or the settings.
%
%      Example 1:
%         s = rng           % get the current generator settings
%         x = rand(1,5)     % RAND generates some values
%         rng(s)            % restore the generator settings
%         y = rand(1,5)     % generate the same values so x and y are equal
% 
%      Example 2:
%         oldS = rng(0,'v5uniform') % use legacy generator
%         x = rand                  % legacy startup value .9501
%         rng(oldS)                 % restore the old settings
%
%   See <a href="matlab:helpview([docroot '\techdoc\math\math.map'],'update_random_number_generator')">Updating Your Random Number Generator Syntax</a> to use RNG to replace
%   RAND or RANDN with the 'seed', 'state', or 'twister' inputs.
%
% MODIFIED BY STEPHEN BECKER
%   See also RAND, RANDI, RANDN, RandStream, NOW.


%   See <a href="http://www.mathworks.com/access/helpdesk/help/techdoc/math/brn4ixh.html#brvku_2">Choosing a Random Number Generator</a> for details on these generators.

%   Copyright 2010 The MathWorks, Inc. 
%   $Revision: 1.1.6.1 $  $Date: 2010/10/25 16:06:38 $

persistent do_once
% 2014, rng is not builtin, it's in a package, so be careful:
C = which('rng','-all');
if isempty( do_once ), do_once = 0; end
if size(C,1) > 1 && do_once < size(C,1)
    do_once = do_once + 1;
    % add this directory to the very top of the path so it shadows this
    % file...
    addpath(fileparts( C{end} ) )
%     disp('Re-run your code; the path to rng has been fixed');
    [varargout{1:nargout}] = rng( varargin{:} );
    return;
end

if exist('rng','builtin')
    [varargout{1:nargout}] = builtin('rng',varargin{:} );
    return;
end

% if exist('rng','builtin')
%     switch nargin
%         case 0
%             if nargout > 0
%                 settings = builtin('rng');
%             else
%                 builtin('rng');
%             end
%         case 1
%             if nargout > 0
%                 settings = builtin('rng',arg1);
%             else
%                 builtin('rng',arg1);
%             end
%         case 2
%             if nargout > 0
%                 settings = builtin('rng',arg1,arg2);
%             else
%                 builtin('rng',arg1,arg2);
%             end
%     end
%     return;
% end

% -- SRB adding this --
error(nargchk(1,1,nargin));
error(nargoutchk(0,0,nargout));
arg1 = varargin{1};
% For R2008a, this doesn't work... (not sure what earliest version is)
if verLessThan('matlab','7.7')
    randn('state',arg1);
    rand('state',arg1);
elseif verLessThan('matlab','8')
    RandStream.setDefaultStream(RandStream('mt19937ar', 'seed', arg1 ));
else
    RandStream.setGlobalStream(RandStream('mt19937ar', 'seed', arg1 ));
end

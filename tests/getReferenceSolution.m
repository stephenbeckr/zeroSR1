global ZEROSR1ROOT
if exist('ZEROSR1ROOT','var') && ~isempty(ZEROSR1ROOT)
    refDir = fullfile(ZEROSR1ROOT,'tests','reference_solutions');
else
    fprintf('\n\nERROR: cannot find variable ZEROSR1ROOT\n');
    fprintf('This is probably because you did not run setup_zeroSR1\n');
    fprintf('  or you "cleared" variables since then. Please re-run setup-zeroSR1\n');
    error('zeroSR1:cannotFindVariable','Cannot find ZEROSR1ROOT');
end

fileName = fullfile(refDir,[problemName,'.mat']);

if exist(fileName,'file')
    fprintf('Loading reference solution from file\n');
    load(fileName); % loads xRef
else
    % Compute answer
    % Do this in a separate file since otherwise
    % Octave cannot parse this.
    
    if ~exist('cvx_begin','file')
        error('Did not find reference solution nor CVX');
    end
    
    computeReferenceSolution; % makes xRef
    
    % and save to the file
    save(fileName,'xRef');
end
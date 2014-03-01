
refDir = fullfile(ZEROSR1ROOT,'tests','reference_solutions');

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
function x_curr=CS_CoSaMP(u,phi_mat,s0)
% Setting up initial parameters for the CoSaMP:
gamma_curr = u; % gamma_0
% x_curr = 0*x; % Initial estimate of x
iter_no = 1;% Initialize the iteration index
% error_iter_main = Inf; % Initialize the error in the iteration to a large no (Infinity here), as it will be updated later

% Stopping threshold for error
% if norm_n_eta > 0
%     thresh = norm_n_eta;
% else
% thresh = 10^-10; 
% end

T_col_set = []; % Initialize the set of columns that are a support for the signal proxy
sup_col_now = []; % Initialize the suport of the current estimate of x
max_iter = 50; % Maximum number of iterations
err_vec = []; % Intialize the vector of errors across iterations
sparsity_est = s0; % The sparsity of the required answer ( e.g.,  we can have a 12 sparse estimate of a 15 sparse vector) 
s0 = sparsity_est; % Setting the sparsity of the required answer to the one specified above

while(iter_no < max_iter) % Check for stopping criteria
    iter_no = iter_no + 1; % Update iteration
    
    y = phi_mat'*gamma_curr; % Form signal proxy
    
    [~, col_now] = sort(abs(y), 'descend'); % Sort the column using absolute values
    col_now = col_now(1:2*s0); % Find the 2*s0 most prominent components of the proxy
    col_now = col_now(:); % Making the set a column vector for consistency in the calculations below
    
    T_col_set = unique([sup_col_now; col_now]); % Merging the supports (in indices of the columns)
    
    % Finding the basis set of the merged support
    T_set =[];
    for col=1:length(T_col_set)
        T_set = [T_set phi_mat(:, T_col_set(col))];
    end
    
    % Stopping criteria, to see if any new data is added, if not then exit
    % the loop
    if length(sup_col_now) == length(T_col_set)
        break
    end
    
    % Finding the estimate of x in reduced dimension using the basis of the
    % merged support
    x_reduced_dim = pinv(T_set)*u;
    
    % Finding the s0 most promininent components of the reduced
    % dimension estimate of x, which will lead to the s0 approximation of x
    [~, pos] = sort(abs(x_reduced_dim), 'descend');
    x_curr = zeros(4096,1);
    for pos_no = 1:s0
        x_curr(T_col_set(pos(pos_no))) = x_reduced_dim(pos(pos_no));
    end
    
    % Updating the column support (sparsity s0) of the current estimate of x
    [~ ,sup_col_now] = sort(abs(x_curr), 'descend');
    sup_col_now = sup_col_now(1:s0);
    
    % Updating the residual
    gamma_curr = u-phi_mat*x_curr;
    
    % Updating the error vector
%     err_vec = [ err_vec norm(x_curr - x)];
    
    % Checking for the stopping criteria
    if norm(gamma_curr) <  10^-10
        break
    end   
    
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the DCT basis is selected as the sparse representation dictionary
% instead of seting the whole image as a vector, I process the image in the
% fashion of column-by-column, so as to reduce the complexity.

% Author: Chengfu Huo, roy@mail.ustc.edu.cn, http://home.ustc.edu.cn/~roy
% Reference: J. Tropp and A. Gilbert, “Signal Recovery from Random 
% Measurements via Orthogonal Matching Pursuit,” 2007.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%************************************************************************%
function hat_x=cs_omp1(y,T_Mat,m,s)
% y=T_Mat*x, T_Mat is n-by-m
% y - measurements
% T_Mat - combination of random matrix and sparse representation basis
% m - size of the original signal
% the sparsity is length(y)/4

n=length(y);
% s=floor(n/4);                                     % Dimension of measured value
hat_x=zeros(1,m);                                 %  Spectral domain (transform domain) vector to be reconstructed                    
Aug_t=[];                                         %  Incremental matrix (initial value is empty matrix)
r_n=y;                                            %  Residual value

for times=1:s                                  %  The number of iterations (the sparsity is measured by 1 hand 4)

    product=abs(T_Mat'*r_n);
    
    [val,pos]=max(product);                       %  Position corresponding to the maximum projection coefficient
    Aug_t=[Aug_t,T_Mat(:,pos)];                   %  Matrix extension
    T_Mat(:,pos)=zeros(n,1);                      %  The selected column is zeroed (essentially removed, for simple zeroing)
    aug_x=(Aug_t'*Aug_t)^(-1)*Aug_t'*y;           %  Least squares to minimize residuals
    r_n=y-Aug_t*aug_x;                            %  Residual error
    pos_array(times)=pos;                         %  Record the position of the maximum projection coefficient
    
end
hat_x(pos_array)=aug_x;                           %  Reconstructed vector
end



%%
%  The function of CoSaMP
%  s-Measured value；T-Observation matrix；N-Vector size
function [ theta,erro_res ] = SP( y,A,K )
    [m,n] = size(y);
    if m<n
        y = y'; 
    end
    [M,N] = size(A); 
    theta = zeros(N,1); 
    pos_num = []; 
    res = y; 
    for kk=1:K 
        %(1) Identification
        product = A'*res; %传感矩阵A各列与残差的内积
        [val,pos]=sort(abs(product),'descend');
        Js = pos(1:K); %选出内积值最大的2K列
        %(2) Support Merger
        Is = union(pos_num,Js); %Pos_theta与Js并集
        %(3) Estimation
        %At的行数要大于列数，此为最小二乘的基础(列线性无关)
        if length(Is)<=M
            At = A(:,Is); %将A的这几列组成矩阵At
        else %At的列数大于行数，列必为线性相关的,At'*At将不可逆
            if kk == 1
                theta_ls = 0;
            end
            break; %跳出for循环
        end
        %y=At*theta，以下求theta的最小二乘解(Least Square)
        theta_ls = (At'*At)^(-1)*At'*y; %最小二乘解
        %(4) Pruning
        [val,pos]=sort(abs(theta_ls),'descend');
        %(5) Sample Update
        pos_num = Is(pos(1:K));
        theta_ls = theta_ls(pos(1:K));
        %At(:,pos(1:K))*theta_ls是y在At(:,pos(1:K))列空间上的正交投影
        res = y - At(:,pos(1:K))*theta_ls; %更新残差 
        erro_res(kk)=norm(res,2);
        if norm(res)<1e-6 %Repeat the steps until r=0
            break; %跳出for循环
        end
    end
    theta(pos_num)=theta_ls; %恢复出的theta
end
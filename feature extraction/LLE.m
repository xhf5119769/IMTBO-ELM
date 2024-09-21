clc
clear
load('data_X.mat')
load('data_Y.mat')
X=X';
D = size(X, 1); % 特征数
N = size(X, 2); % 样本数
K = 50; % 近邻的数量
d = 11 ;  % 降维后的目标维度
% 计算距离矩阵
distanceMatrix = squareform(pdist(X', 'euclidean'));
% 找到每个点的K个最近邻
[~, neighbors] = sort(distanceMatrix, 2, 'ascend');
neighbors = neighbors(:, 2:K+1); % 排除自身
W = zeros(N, N); % 初始化重建权重矩阵
for i = 1:N
    Z = X(:, neighbors(i, :)) - repmat(X(:, i), 1, K); % 减去本身，中心化
    C = Z' * Z; % 局部协方差
    C = C + eye(K, K) * 1e-3 * trace(C); % 正则化防止奇异
    w = C \ ones(K, 1); % 解线性系统
    W(i, neighbors(i, :)) = w' / sum(w); % 归一化并赋值
end
M = (eye(N) - W)' * (eye(N) - W); % 计算矩阵M
options.disp = 0; % 不显示额外输出
options.isreal = 1; % 实数
options.issym = 1; % 对称
[X_L, ~] = eigs(M, d + 1, 'sm', options); % 计算最小的d+1个特征值和对应的特征向量
X_LLE = X_L(:, 2:d+1)'; % 排除第一个特征向量，转置以便每行是一个维度

Y=Y1';
R2=[];
RMSE=[];
MAPE=[];
MAE=[];
CCC=[];
R=[];
for i=1:10
QQ{i}=randperm(size(X_LLE,2));
end
for i=1:10
  X_train{i}=X_LLE(:,QQ{i}(1:300));
  y_train{i}=Y(:,QQ{i}(1:300));
  X_test{i}=X_LLE(:,QQ{i}(301:end));
  y_test{i}=Y(:,QQ{i}(301:end));
 
  N=20;%%隐含层节点
  [IW,B,LW,TF,TYPE] = elmtrain(X_train{i},y_train{i},N,'sig');
  T{i}= elmpredict(X_test{i},IW,B,LW,TF,TYPE);
    L=length(T{i});
    %% 指标计算
    SS_tot = sum((y_test{i} - mean(y_test{i})).^2);
    % 计算残差平方和（SS_res）
    SS_res = sum((y_test{i} - T{i}).^2);
    % 计算决定系数（R^2）
    R2(i) = 1 - (SS_res / SS_tot);
    RMSE(i)=sqrt(sum((y_test{i}-T{i}).^2)/L);
    MAPE(i)=mean(abs((y_test{i} - T{i})/y_test{i}))*100;
    MAE(i)=mean(abs(y_test{i} - T{i}));
    R(i)=abs((L*sum(T{i}.*y_test{i})-sum(T{i})*sum(y_test{i}))/sqrt(((L*sum((T{i}).^2)-(sum(T{i}))^2)*(L*sum((y_test{i}).^2)-(sum(y_test{i}))^2))));
   mu_x(i) = mean(y_test{i}); % 真实值均值
   mu_y(i) = mean(T{i}); % 预测值均值
   sigma_x(i) = std(y_test{i}); % 真实值标准差
   sigma_y(i) = std( T{i}); % 预测值标准差
   CCC(i) = (2 * R(i) * sigma_x(i) * sigma_y(i)) / (sigma_x(i)^2 + sigma_y(i)^2 + (mu_x(i) - mu_y(i))^2);
end
R2_final=mean(R2)
RMSE_final=mean(RMSE)
MAPE_final=mean(MAPE)
MAE_final=mean(MAE)
CCC_final=mean(CCC)
R_final=mean(R)
MV=[R2_final RMSE_final MAPE_final MAE_final CCC_final R_final];
MV_D=[1 0 0 0 1 1];
SRD=sum(abs(MV_D-MV))

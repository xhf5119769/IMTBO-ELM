clc
clear
load('data_Y.mat')
load('data_X.mat')
X=X';
[icasig, A, W] = fastica(X);
ICA_data=icasig;
for i=1:10
QQ{i}=randperm(size(ICA_data,2));
end
pca_data=ICA_data;
Y=Y1';
R2=[];
RMSE=[];
MAPE=[];
MAE=[];
CCC=[];
R=[];


for i=1:10
    X_train{i}=pca_data(:,QQ{i}(1:300));
    y_train{i}=Y(:,QQ{i}(1:300));
    X_test{i}=pca_data(:,QQ{i}(301:end));
    y_test{i}=Y(:,QQ{i}(301:end));
    %% 归一化
    [Xn_train{i},in]=mapminmax(X_train{i});
    [yn_train{i},ou]=mapminmax(y_train{i});
    Xn_test{i}=mapminmax('apply',X_test{i}, in);
    N=100;%%隐含层节点
    [IW,B,LW,TF,TYPE] = elmtrain(Xn_train{i},yn_train{i},N,'sig');
    Tn_test{i} = elmpredict(Xn_test{i},IW,B,LW,TF,TYPE);
    T{i}=mapminmax('reverse',Tn_test{i},ou);
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
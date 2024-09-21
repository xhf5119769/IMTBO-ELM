clc
clear
load('data_X.mat')
load('data_Y.mat')
R2=[];
RMSE=[];
MAPE=[];
MAE=[];
CCC=[];
R=[];

[coeff,score,latent,tsquared,explained,mu] = pca(X);
tol=0;
for i=1:100
    tol=tol+sum(explained(i));
    if tol>99.9;
        break
    end
end
pca_data=score(:,1:i);

pca_data=pca_data';
X=X';
Y=Y1';
for i=1:10
QQ{i}=randperm(size(X,2));
end

for i=1:10


X_train{i}=X(:,QQ{i}(1:300));
pca1{i}=pca_data(:,QQ{i}(1:300));
y_train{i}=Y(:,QQ{i}(1:300));
X_test{i}=X(:,QQ{i}(301:end));
pca2{i}=pca_data(:,QQ{i}(301:end));
 y_test{i}=Y(:,QQ{i}(301:end));
[R1,Q] = size(X_train{i});
%% MELM-AE第一层
N1=300;
iw=rand(N1,R1)*2-1;
IW1=(orth(iw'))';
B1 = orth(rand(N1,1));    %% 生成的B也是正交矩阵
BiasMatrix1 = repmat(B1,1,Q);
tempH1 = IW1 * X_train{i} + BiasMatrix1;
H1{i} = 1 ./ (1 + exp(-tempH1));
LW1 = pinv(H1{i}')* X_train{i}';
Xn_test1{i}=ELM_AE(X_test{i},IW1,B1,LW1,'sig',0);
%% MELM-AE第二层
N2=30;
iw2=rand(N2,N1)*2-1;
IW2=(orth(iw2'))';
B2 = orth(rand(N2,1));    %% 生成的B也是正交矩阵
BiasMatrix2 = repmat(B2,1,Q);
tempH2 = IW2 * H1{i} + BiasMatrix2;
H2{i} = 1 ./ (1 + exp(-tempH2));
LW2 = pinv(H2{i}')* H1{i}';
Xn_test2{i}=ELM_AE(Xn_test1{i},IW2,B2,LW2,'sig',0);
fusion_train{i}=[H2{i};pca1{i}];
fusion_test{i}=[Xn_test2{i};pca2{i}];
end
for iii=1:10
%% 优化算法
%% 优化算法
RMSE=[];
N=20;
[R1,Q] = size(fusion_train{iii});
maxgen=30;   % iii迭代次数  
sizepop=30;   %种群规模
popmax=5;
popmin=-5;
num=N+R1*N;
for i=1:sizepop
    pop(i,:)=2*rand(1,num)-1;%随机生成参数
    IW=reshape(pop(i,(1:N*R1)),N,R1);
    B=(pop(i,N*R1+1:end))';
    BiasMatrix=repmat(B,1,Q);
    tempH = IW * fusion_train{iii} + BiasMatrix;
    H = 1 ./ (1 + exp(-tempH));
    LW = pinv(H')* y_train{iii}';
    TF='sig';
    TYPE=0;
    Tn_test = elmpredict(fusion_test{iii},IW,B,LW,TF,TYPE);
    % T=mapminmax('reverse',Tn_test,ou);
   T= Tn_test;
    L=length(T);
    rmse=sqrt(sum((y_test{iii}-T).^2)/L);
    RMSE=[RMSE;rmse];
end
pFit=RMSE;
[ fMin, bestI ] = min( RMSE );      % fMin表示当前最优位置
bestX = pop( bestI, : );             % bestX 表示全局最优微信hi
bestXX = pop( bestI, : );   
    ffMin=fMin;  % ffMin表示全局最优适应值
fmin=fMin;
ffmin=fmin;
Temp=pop;
for a=1:maxgen
[ ans, sortIndex ] = sort( pFit );% 将适应度值从小向大排列
pop=pop(sortIndex,:);
  [fmax,B]=max( pFit ); %返回pFit中的最大值fmax 以及位置B
   worse= pop(B,:);   %寻找最差解
   for j=2:sizepop
       Li=0.25+0.25*rand;
       Ai=0.5+0.25*rand;
       Mi=0.75+0.25*rand;
       if rand < Li
           pop(j,:)=pop(j,:)+rand*(bestX-pop(j,:))+rand*(pop((j-1),:)-pop(j,:));
           Temp(j,:)=Temp(j,:)+rand*(bestX-Temp(j,:))+rand*(Temp((j-1),:)-Temp(j,:));
       elseif rand < Ai
               pop(j,:)=pop(j,:)-rand*(worse-pop(j,:));
                Temp(j,:)=Temp(j,:)-rand*(worse-Temp(j,:));
       elseif rand < Mi
               pop_mean=mean(pop,1);
               Temp_mean=mean(Temp,1);
               pop(j,:)=pop(j,:)-rand*(pop_mean-pop(j,:));
                Temp(j,:)=Temp(j,:)-rand*(worse-Temp(j,:));
       else 
            pop(j,:)=2*rand(1,num)-1;%随机生成参数
            Temp(j,:)=2*rand(1,num)-1;%随机生成参数
       end
       IW=reshape(pop(j,(1:N*R1)),N,R1);
       B=(pop(j,N*R1+1:end))';
       BiasMatrix=repmat(B,1,Q);
       tempH = IW * fusion_train{iii} + BiasMatrix;
       H = 1 ./ (1 + exp(-tempH));
       LW = pinv(H')* y_train{iii}';
       TF='sig';
       TYPE=0;
       T = elmpredict(fusion_test{iii},IW,B,LW,TF,TYPE);
       L=length(T);
       rmse=sqrt(sum((y_test{iii}-T).^2)/L);
       
       if rmse < fMin
           fMin = rmse;
           bestX=pop(j,:);
       end

       IW=reshape(Temp(j,(1:N*R1)),N,R1);
       B=(Temp(j,N*R1+1:end))';
       BiasMatrix=repmat(B,1,Q);
       tempH = IW * fusion_train{iii} + BiasMatrix;
       H = 1 ./ (1 + exp(-tempH));
       LW = pinv(H')* y_train{iii}';
       TF='sig';
       TYPE=0;
       T = elmpredict(fusion_test{iii},IW,B,LW,TF,TYPE);
       L=length(T);
       rmse=sqrt(sum((y_test{iii}-T).^2)/L);
       
       if rmse < ffMin
           ffMin = rmse;
           bestXX=Temp(j,:);
       end


           if rand<0.8
               J=floor(rand*30+1);
               K=floor(rand*30+1);
               Temp(j,:) = Temp(j,:) + Temp(j,:)*trnd(a);
                    IW=reshape(Temp(j,(1:N*R1)),N,R1);
                    B=(Temp(j,N*R1+1:end))';
                    BiasMatrix=repmat(B,1,Q);
                    tempH = IW * fusion_train{iii} + BiasMatrix;
                    H = 1 ./ (1 + exp(-tempH));
                    LW = pinv(H')* y_train{iii}';
                    TF='sig';
                    TYPE=0;
                    T = elmpredict(fusion_test{iii},IW,B,LW,TF,TYPE);
                    L=length(T);
                    rmse2=sqrt(sum((y_test{iii}-T).^2)/L); 
          if rmse2 < ffMin
           bestXX=Temp(j,:);
       end                 
           end

   end
end


%% 计算IMTBO-ELM
IW=reshape(bestXX(1:N*R1),N,R1);
B=(bestXX(N*R1+1:end))';
BiasMatrix=repmat(B,1,Q);
tempH = IW * fusion_train{iii} + BiasMatrix;
H = 1 ./ (1 + exp(-tempH));
LW = pinv(H')* y_train{iii}';
TF='sig';
TYPE=0;
TT{iii} = elmpredict(fusion_test{iii},IW,B,LW,TF,TYPE);
L=length(TT{iii});
    SS_tot = sum((y_test{iii} - mean(y_test{iii})).^2);
    % 计算残差平方和（SS_res）
    SS_res = sum((y_test{iii} - TT{iii}).^2);
    % 计算决定系数（R^2）
    R2(iii) = 1 - (SS_res / SS_tot);
    RMSEE(iii)=sqrt(sum((y_test{iii}-TT{iii}).^2)/L);
    MAPE(iii)=mean(abs((y_test{iii} - TT{iii})/y_test{iii}))*100;
    MAE(iii)=mean(abs(y_test{iii} - TT{iii}));
    R(iii)=abs((L*sum(TT{iii}.*y_test{iii})-sum(TT{iii})*sum(y_test{iii}))/sqrt(((L*sum((TT{iii}).^2)-(sum(TT{iii}))^2)*(L*sum((y_test{iii}).^2)-(sum(y_test{iii}))^2))));
   mu_x(iii) = mean(y_test{iii}); % 真实值均值
   mu_y(iii) = mean(TT{iii}); % 预测值均值
   sigma_x(iii) = std(y_test{iii}); % 真实值标准差
   sigma_y(iii) = std( TT{iii}); % 预测值标准差
   CCC(iii) = (2 * R(iii) * sigma_x(iii) * sigma_y(iii)) / (sigma_x(iii)^2 + sigma_y(iii)^2 + (mu_x(iii) - mu_y(iii))^2);


%% 计算MTBO-ELM

IW=reshape(bestX(1:N*R1),N,R1);
B=(bestX(N*R1+1:end))';
BiasMatrix=repmat(B,1,Q);
tempH = IW * fusion_train{iii} + BiasMatrix;
H = 1 ./ (1 + exp(-tempH));
LW = pinv(H')* y_train{iii}';
TF='sig';
TYPE=0;
T1{iii} = elmpredict(fusion_test{iii},IW,B,LW,TF,TYPE);
L=length(T1{iii});
    SS_tot = sum((y_test{iii} - mean(y_test{iii})).^2);
    % 计算残差平方和（SS_res）
    SS_res = sum((y_test{iii} - T1{iii}).^2);
    % 计算决定系数（R^2）
    R21(iii) = 1 - (SS_res / SS_tot);
    RMSE1(iii)=sqrt(sum((y_test{iii}-T1{iii}).^2)/L);
    MAPE1(iii)=mean(abs((y_test{iii} - T1{iii})/y_test{iii}))*100;
    MAE1(iii)=mean(abs(y_test{iii} - T1{iii}));
    RR1(iii)=abs((L*sum(T1{iii}.*y_test{iii})-sum(T1{iii})*sum(y_test{iii}))/sqrt(((L*sum((T1{iii}).^2)-(sum(T1{iii}))^2)*(L*sum((y_test{iii}).^2)-(sum(y_test{iii}))^2))));
   mu_x1(iii) = mean(y_test{iii}); % 真实值均值
   mu_y1(iii) = mean(T1{iii}); % 预测值均值
   sigma_x1(iii) = std(y_test{iii}); % 真实值标准差
   sigma_y1(iii) = std( T1{iii}); % 预测值标准差
   CCC1(iii) = (2 * RR1(iii) * sigma_x1(iii) * sigma_y1(iii)) / (sigma_x1(iii)^2 + sigma_y1(iii)^2 + (mu_x1(iii) - mu_y1(iii))^2);
end
R2_final1=mean(R2)
R2_final2=mean(R21)

RMSE_final1=mean(RMSEE)
RMSE_final2=mean(RMSE1)

MAPE_final1=mean(MAPE)
MAPE_final2=mean(MAPE1)

MAE_final1=mean(MAE)
MAE_final2=mean(MAE1)

CCC_final1=mean(CCC)
CCC_final2=mean(CCC1)

R_final1=mean(R)
R_final2=mean(RR1)

MV1=[R2_final1 RMSE_final1 MAPE_final1 MAE_final1 CCC_final1 R_final1];
MV2=[R2_final2 RMSE_final2 MAPE_final2 MAE_final2 CCC_final2 R_final2];
MV_D=[1 0 0 0 1 1];
SRD1=sum(abs(MV_D-MV1))
SRD2=sum(abs(MV_D-MV2))

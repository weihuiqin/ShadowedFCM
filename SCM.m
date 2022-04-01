function [U,P,Dist,Cluster_Res,Obj_Fcn,iter,val]=SCM(Data,P0,plotflag,M,epsm)
% Fuzzy c-means clustering (FCM) : began to iterate
% from the initial clustering center
% [U,P,Dist,Cluster_Res,Obj_Fcn,iter] = fuzzycm2(Data,P0,plotflag,M,epsm)
% inout: Data,plotflag,M,epsm: see fuzzycm.m
% P0:initial clustering center
% output: U,P,Dist,Cluster_Res,Obj_Fcn,iter: see fuzzycm.m
% See also: fuzzycm
if nargin<5
    epsm=1.0e-6;
end
if nargin<4
    M=2;
end
if nargin<3
    plotflag=0; 
end
[N,S] = size(Data); m = 2/(M-1); iter = 0;
C=size(P0,1);Dist=zeros(C,N);U=zeros(C,N);P=zeros(C,S);
% The iteration algorithm of FCM
while true
    % Iterative counter
    iter=iter+1;
    fprintf('*');
    % Calculate or update divided matrix U   
    if size(P0, 2) > 1,
        for i=1:C
%                     for j=1:N
%                         Dist(i,j)=fuzzydist(P0(i,:),Data(j,:));
%                     end
           Dist(i,:)=sqrt(sum(((Data-ones(size(Data, 1), 1)*P0(i,:)).^2)'));
        end
    else
        for i=1:C
            Dist(i, :) = abs(P0(i)-Data)';
        end
    end
    U=1./(Dist.^m.*(ones(C,1)*sum(Dist.^(-m))));
    U(Dist==0)=1;

	%%%%
    I=findSplitPoint(U);
    Umax= max(U,[],2);
    %%%%
	
    P=zeros(C,S);
    for i=1:C
        a=0;
        b=0;
        c=0;
        for j=1:N
            if U(i,j)>=(Umax(i)-I(i,1))
                P(i,:)=P(i,:)+ Data(j,:);
                a=a+1;
            end
            if U(i,j)>I(i,1)&& U(i,j)<(Umax(i)-I(i,1))
                P(i,:)=P(i,:)+ (U(i,j).^M).*Data(j,:);
                b=b+U(i,j).^M;
            end
            if U(i,j)<=I(i,1)
                P(i,:)=P(i,:)+ (U(i,j).^(M.^M)).*Data(j,:);
                c=c+(U(i,j).^(M.^M));
            end
        end
        P(i,:)=P(i,:)./(a+b+c);
    end
    Um=U.^M;
    %P=Um*Data./(ones(S,1)*sum(Um'))';
    % the objective function value: intra-class sum of weighted square error
    if nargout>4 || plotflag
        Obj_Fcn(iter)=sum(sum(Um.*Dist.^2));
    end
    % iterative stop condition of FCM algorithm
    if norm(P-P0,Inf)<epsm || iter>100
        %fprintf('Obj_Fcn=%d,', Obj_Fcn(iter));
        break
    end
    P0=P;
end
fprintf('\n');

%% the DBI(davies-bouldin)
% dw=zeros(C,C);
% for k=1:C
%     for l=1:C
%         if k==l
%             dw(l,k)=0;
%         end
%         if k~=l
%             dw(k,l)=(sum(Dist(k,:))+sum(Dist(l,:)))./fuzzydist(P(k,:),P(l,:));
%         end
%     end
% end
% db=sum(max(dw,[],2))./C;
% fprintf('the DBI=%d,', db);
% %The xie-beni index
% d=zeros(C,C);
% for i=1:C
%     for j=1:C
%         if i==j
%           d(i,j)=1000000;
%         end
%         if i~=j
%           d(i,j)=fuzzydist(P(i,:),P(j,:));
%         end
%     end
% end
% dmin=min(min(d));   
% xie=Obj_Fcn(iter)./(N*dmin);
% fprintf('xie-beni index=%d\n', xie);
%% Clustering results
if nargout > 3
    res = maxrowf(U);
    for c = 1:C
        v = find(res==c);
        Cluster_Res(c,1:length(v))=v;
    end
end
% plot
if plotflag
    Sfcmplot(Data,U,I,P,Obj_Fcn);
end
%validity
myval1=myvality(U,Data,P,Dist.^2,1);
myval2=myvality(U,Data,P,Dist.^2,2);
myval3=myvality(U,Data,P,Dist.^2,3);
val.PC=myval1.validity.PC;
val.CE=myval1.validity.CE;
val.SC=myval2.validity.SC;
val.S=myval2.validity.S;
val.XB=myval2.validity.XB;
val.DI=myval3.validity.DI;
val.ADI=myval3.validity.ADI;


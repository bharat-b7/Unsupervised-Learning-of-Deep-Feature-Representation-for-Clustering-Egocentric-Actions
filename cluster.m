clc;clear;
%% Set variables
path1='./gteagroundtruth/';
suffix='lstm_ver4_2';
d1=dir(['./folder5/S2/S2*' suffix]);
d2=dir(['./folder5/S2_frame/S2*' suffix]);
d3=dir('./gteagroundtruth/*mat');
no_classes=11;
confuse=zeros(no_classes);
plo=cell(1,length(d1));
for k=1:length(d1)
    x=readNPY(['./folder5/S2/' d1(k).name '/feat_x.npy']);
    y=readNPY(['./folder5/S2/' d1(k).name '/feat_y.npy']);
    z=readNPY(['./folder5/S2_frame/' d2(k).name '/feat_.npy']);
    GT=load(['./gteagroundtruth/' d3(k).name]);GT=GT.fin;
    GT=GT(1:size(x,1));
    bow=[x,y,z];
    f1=[];best=[];bmap=[];
    parfor i=1:400
        classes2=kmeans(bow,length(unique(GT)));
        [map,fin_f1]=match_greedy2(GT',classes2);
        f1=[f1;fin_f1];
        bmap=[bmap;map];
        best=[best;classes2'];
    end
    [~,in]=max(f1);
    h=best(in,:);
    q=bmap(2*in-1:2*in,:);
    rep=zeros(size(h));
    for i=1:size(q,2)
        rep(h==q(2,i))=q(1,i);
    end
    CM = confusion(GT',rep,no_classes);
    confuse=confuse+CM;
    plo{k}=[GT';rep];
    disp(k)
end

sum(diag(confuse))/sum(confuse(:))
clc;clear;
%% Set variables
path1='./gteagroundtruth/';
pre={'S2'};

sp_len=2;%% in sec.
fr_rate=15;%% in fps.
%%
lab=[];
for j=1:length(pre)
    d=load(['metadata_' pre{j} '.mat']);
    d=d.d;
    for i=1:length(d)
        fid=fopen([path1 d(i).name],'r');
        a=fscanf(fid,'%c');
        fclose(fid);
        tokens = strsplit(a);
        if(strcmp(tokens(end),''))
            tokens(end)=[];
        end
        tokens=reshape(tokens,4,length(tokens)/4)';
        tokens(strcmp(tokens(:,1),'x'),:)=[];
        unlab=unique(tokens(:,1));
        lab=[lab;unlab];
    end
end
unlab=unique(lab);
comb_GT=[];bounds=0;
for o=1:length(pre)
    d=load(['metadata_' pre{o} '.mat']);
    d=d.d;
    comb_GT=[];
    for i=1:length(d)
        GT=-1*ones(floor(d(i).Duration*fr_rate),1);
        fid=fopen([path1 d(i).name],'r');
        a=fscanf(fid,'%c');
        fclose(fid);
        tokens = strsplit(a);
        if(strcmp(tokens(end),''))
            tokens(end)=[];
        end
        tokens=reshape(tokens,4,length(tokens)/4)';
        tokens(strcmp(tokens(:,1),'x'),:)=[];
        for j=1:size(tokens,1)
            ts=str2double(tokens{j,3});te=str2double(tokens{j,4});
                GT(ts:te)=find(strcmp(unlab,tokens(j)));
        end
        GT=GT(1:length(GT)-mod(length(GT),30));
        fin=zeros(floor(length(GT)/sp_len/fr_rate),1);
        for j=1:floor(length(GT)/sp_len/fr_rate)
            fin(j)=mode(GT((j-1)*sp_len*fr_rate+1:j*sp_len*fr_rate));
        end
    %     save([path1 d(i).name(1:end-4) '_GT.mat'],'fin')
%         save([path1 d(i).name(1:end-4) '_GT_frame.mat'],'GT')
        fin(fin==-1)=0;
        fin=fin+1;
        bounds=[bounds,bounds(end)+length(fin)];
        save([path1 d(i).name(1:end-4) '_GT.mat'],'fin')
    end
end
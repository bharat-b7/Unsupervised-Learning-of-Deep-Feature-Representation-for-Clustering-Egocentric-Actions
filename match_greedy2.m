function [map,fin_f1]=match_greedy2(GT,B)
unGT=unique(GT);unB=unique(B);n=0;fin_f1=0;map=[];
[~,in]=sort(hist(GT,unGT),'descend');
unGT=unGT(in);
% unGT(unGT==1)=[];unGT=[unGT,1];
while(1)
    m=-1;fi=-1;fj=-1;
    for i=1:length(unGT)
        if unGT(i)==-1 
            continue; 
        end
        for j=1:length(unB)
            if unB(j)==-1 
                continue;
            end
            q=find(B==unB(j)); w=find(GT==unGT(i));
            x=length(intersect(q,w));
            pre=x/length(q); rec=x/length(w);
            if (pre+rec) ~=0
                f1=2*pre*rec/(pre+rec);
            else f1=0;
            end
            cost=f1;
            if cost>m
                m=cost; fi=i; fj=j; maxx=length(w); fin=f1;
            end
        end
    end
    map=[map,[unGT(fi);unB(fj)]];
    unGT(fi)=-1;unB(fj)=-1;
    fin_f1=(fin_f1*n+fin*maxx)/(n+maxx);
    n=n+maxx;
    if all(unB==-1) || all(unGT==-1)
%     if all(unGT==-1)
        break
    end
end
end
function ret=ind2vec(x,n)
    ret=zeros(size(x,1),n);
    u=unique(x);
    for i=1:length(u)
        ret(x==u(i),i)=1;
    end
end
function ret=confusion(x,y,n)
    ret=zeros(n);
    for i=1:length(x)
        ret(x(i),y(i))=ret(x(i),y(i))+1;
    end
end
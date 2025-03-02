function Y=hessNum(e,A,Xx,m)
[~,n]=size(A);
Y=zeros(size(Xx,1));
for j=1:(n+1)*m
    h=1e-5;
    Xx1=Xx;
    Xx1(j)=Xx1(j)-h;
    Xx2=Xx;
    Xx2(j)=Xx2(j)+h;
    Y(:,j)=(gradNum(e,A,Xx2,m)-gradNum(e,A,Xx1,m))/2/h;
end
end

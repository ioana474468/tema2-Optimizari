function y=gradNum(e,A,Xx,m)
[~,n]=size(A);
y=zeros(size(Xx));
for i=1:(n+1)*m
    h=1e-5;
    Xx1=Xx;
    Xx1(i)=Xx1(i)-h;
    Xx2=Xx;
    Xx2(i)=Xx2(i)+h;
    y(i)=(L(e,A,Xx2,m)-L(e,A,Xx1,m))/2/h;
end
end

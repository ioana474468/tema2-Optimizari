function y=L(e,A,Xx,m)
[N,n]=size(A);
X=reshape(Xx(1:n*m),m,n)';
x=Xx(n*m+1:(n+1)*m);
y=(norm(g(A*X)*x-e))^2/2/N;
end

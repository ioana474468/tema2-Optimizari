clear
A=readmatrix("Real estate valuation data set.xlsx");
A=(A-mean(A))./std(A);
A_train=A(1:331,2:7);
A_test=A(332:414,2:7);
e_train=A(1:331,8);
e_test=A(332:414,8);
[N_train,n]=size(A_train);
N_test=size(A_test,1);

A_train=[A_train ones(N_train,1)];
A_test=[A_test ones(N_test,1)];
m=10;

%% metoda gradient

iter=0;
iterMax=200;
eps=0.01; 

Xx=randn((n+2)*m,1);

tic
while iter==0 || (fOb(iter)>eps && iter<iterMax)
    gradient=gradNum(e_train,A_train,Xx,m);
    alfaK=fminbnd(@(alfa) L(e_train,A_train,Xx-alfa*gradient,m),0,0.05);
    Xx=Xx-alfaK*gradient;
    iter=iter+1;
    normG(iter)=norm(gradient);
    fOb(iter)=L(e_train,A_train,Xx,m);
    v(iter)=toc;
end

semilogy(1:iter,fOb)
hold on
semilogy(1:iter,normG)
xlabel("Iterație")
legend("Funcția obiectiv","Norma gradient")
title("Metoda Gradient în funcție de iterație")
hold off

semilogy(v,fOb)
hold on 
semilogy(v,normG)
xlabel("timp")
legend("Funcția obiectiv","Norma gradient")
title("Metoda Gradient în funcție de timp")
hold off

X=reshape(Xx(1:(n+1)*m),m,n+1)';
x=Xx((n+1)*m+1:(n+2)*m);

% scor de antrenare
y_train=g(A_train*X)*x;
R2_train=1-(norm(e_train-y_train))^2/(norm(e_train-mean(e_train)))^2

% scor de testare
y_test=g(A_test*X)*x;
R2_test=1-(norm(e_test-y_test))^2/(norm(e_test-mean(e_test)))^2

%% metoda newton
iter=0;
iterMax=200;
eps=0.01; 

Xx=randn((n+2)*m,1);

tic
while iter==0 || (fOb(iter)>eps && iter<iterMax)
    gradient=gradNum(e_train,A_train,Xx,m);
    d=hessNum(e_train,A_train,Xx,m)\gradient;
    alfaK=fminbnd(@(alfa) L(e_train,A_train,Xx-alfa*d,m),0,0.05);
    Xx=Xx-alfaK*d;
    iter=iter+1;
    normG(iter)=norm(gradient);
    fOb(iter)=L(e_train,A_train,Xx,m);
    v(iter)=toc;
end

semilogy(1:iter,fOb)
hold on
semilogy(1:iter,normG)
xlabel("Iterație")
legend("Funcția obiectiv","Norma gradient")
title("Metoda Newton în funcție de iterație")
hold off

semilogy(v,fOb)
hold on 
semilogy(v,normG)
xlabel("timp")
legend("Funcția obiectiv","Norma gradient")
title("Metoda Newton în funcție de timp")

X=reshape(Xx(1:(n+1)*m),m,n+1)';
x=Xx((n+1)*m+1:(n+2)*m);

% scor de antrenare
y_train=g(A_train*X)*x;
R2_train=1-(norm(e_train-y_train))^2/(norm(e_train-mean(e_train)))^2

% scor de testare
y_test=g(A_test*X)*x;
R2_test=1-(norm(e_test-y_test))^2/(norm(e_test-mean(e_test)))^2


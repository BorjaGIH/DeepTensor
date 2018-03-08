function f1 = f1(x)
%     f1 = x(1)*x(2)+2*x(2)*x(4)*x(3)+5*x(3)+x(4); % linear problem 1st order interaction
    f1 = x(1)^2*x(2)+2*x(2)*x(4)*x(3)+5*x(3)+x(4)^3; % nonlinear problem 2nd order interaction
end
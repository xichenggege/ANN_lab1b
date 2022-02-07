function [W,V,error1,error2] = backprob_MLP(patterns,T,eta,w,v)
    % 2.2.1 The forward pass
    % Two layers neural network
    % hidden layer

    ndata = length(patterns(1,:));

%     hin  = w * [patterns; ones(1,ndata)];
%     hout = [2./(1+exp(-hin))-1; ones(1,ndata)]; 
        hin  = w * [patterns];
    hout = [2./(1+exp(-hin))-1]; 

    % output layer
    oin  = v * hout;
    out  = 2./(1+exp(-oin))-1;
    
    %2.2.2 The backward pass
    % weight update
    delta_o = (out-targets).*((1+out).*(1-out))*(1/2);
    delta_h = (v'*delta_o).*((1+hout).*(1-hout))*(1/2);
    delta_h = delta_h(1:Nhidden,:);   % remove bias
    
    % update weight with momentum term
    alpha = 0.9;  % controls updating fraction
    eta   = 0.01; % learning rate
    dw = (dw.*alpha) - (delta_h * patterns').*(1-alpha);
    dv = (dv.*alpha) - (detra_o * hout').*(1-alpha);
    W  = w + dw.* eta;
    V  = v + dv.* eta;

    % error 1 -> meanSquare error
%     error1 = 
    % error 2 -> number of mismatch
%     error2 =
end
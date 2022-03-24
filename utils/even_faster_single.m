function [o] = even_faster_single(x)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here


    %maxvalue=max(x(:));
    %required_digits = ceil(log(double(maxvalue+1))/log(10));
    required_digits = ceil(log(double(x+1))/log(10));
    %o=repmat(x(1)*0,size(x,1),required_digits);
    %o=repmat(0,1,required_digits);
    o = zeros(1,required_digits);
    for c=required_digits:-1:1
        o(:,c)=mod(x,10);
        x = (x-o(:,c))/10;
    end
    o = char(o+'0');

end


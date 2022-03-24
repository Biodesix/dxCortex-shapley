function [o] = myfastint2str(x)
%UNTITLED Summary of this function goes here
%   x = array of strings

    if (isempty(x))
        o = x;
        return;
    end

    o = even_faster_single(x(1));
    for i =2:length(x)
        o = [o ' ' even_faster_single(x(i))];   % this is faster than strappend
    end

end


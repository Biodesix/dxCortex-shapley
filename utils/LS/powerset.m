function ps = powerset(set)
%powerset
%   calculates the powerset of a set uses cell arrays
%   v1.0        HR                  3/13/2022
    elementlist = 1:1:length(set);
    powersetsize = 2^length(set)-1;
    ps = cell(1,powersetsize);
    i = 0;
    for k = 1:length(elementlist)
        combinations = nchoosek(elementlist,k);
        for j = 1:size(combinations,1)
            i = i + 1;
            ps{i} = set(combinations(j,:));
        end
    end
end


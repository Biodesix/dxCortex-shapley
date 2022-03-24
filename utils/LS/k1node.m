classdef k1node < handle
%k1node 
%   A class containing the features and the mini-classifiers of an
%   element on the first level of the k=1 level structure
%   v1.0        HR                  3/14/2022
    
    properties
        features            % cell array of doubles
        mcs                 % array of mini classifier indices, maybe unnecessary
    end
    
    methods
        function obj = k1node(varargin)
            %Constructor Construct an instance of this class
            %   overload constructor using varargin
            %
            %       length(varargin) == 3: mc_used,ftrs_used_in_kNN, N_ftrs_used_in_kNN
            %           construct from description of mini-classifiers
            %       length(varargin) == 2: mcs, features
            %           construct from list of mini-classifiers features
            %           and their number of elements
            
            if (length(varargin) == 3)
                mc_used             = varargin{1};
                ftrs_used_in_kNN    = varargin{2};
                N_ftrs_used_in_kNN  = varargin{3};
                obj.mcs             = mc_used;
                % for the features use ismember and loop probably faster
                % than union
                ftrs        = zeros(1,2*length(mc_used));
                cnt         = 0;
                for i = 1:length(mc_used)
                    tmp = ftrs_used_in_kNN( mc_used(i),1:N_ftrs_used_in_kNN(mc_used(i)) );
                    for j = 1:length(tmp)
                        if ( ~ismember(tmp(j),ftrs) )
                            cnt         = cnt + 1;
                            ftrs(cnt)   = tmp(j);
                        end
                    end
                end               
                obj.features = num2cell(ftrs(1:cnt));   %convert to cell array
            elseif (length(varargin) == 2)
                obj.mcs         = varargin{1};
                obj.features    = varargin{2};
            else
                error([' wrong number of arguments to k1node constructor ' num2str(length(varargin))])
            end
                       
        end
        
        function dd = add(obj,node)
            %add
            %   Add a k1node to an existing node returns a k1node
            ftrs        = num2cell( union( cell2mat(obj.features),cell2mat(node.features)));
            mc          = union( obj.mcs,node.mcs);
            dd = k1node(mc,ftrs);
            
        end
    end
end


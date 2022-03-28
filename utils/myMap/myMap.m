classdef myMap < handle
    %UNTITLED4 Summary of this class goes here
    %   uses internal data store to provide a hash index to a set of double
    %   arrays and a simple map to do the hashing
    
    properties
        data        % contains the data
        hash        % contains the hashmap
        data_size   % data = array(data_size,...)
        chunk_size  % the number of chunks to prestore
        N           % the number of elements in myMap
        curr_size   % size of space to store
    end
    
    methods
        function obj = myMap(NS,chunksize)
            %UNTITLED4 Construct an instance of this class
            %   Detailed explanation goes here
            obj.data_size   = NS;
            obj.chunk_size  = chunksize;
            
            % make the first chunk
            obj.data = zeros(obj.data_size,chunksize);
            obj.curr_size = chunksize;
            
            %initialize the hash list
            obj.hash    = containers.Map('Keytype','char','ValueType','uint16');
            obj.N       = 0;
        end
        
        function out = isKey(obj,tKey)
            out = obj.hash.isKey(tKey);
        end
        
        function add(obj,key,value)
            if ( obj.hash.isKey(key))
                error(['myMap.add: adding the same key again'])
            else
                % keys 
                obj.N = obj.N + 1;
                obj.hash(key) = obj.N;
                % store data
                if (obj.N > obj.curr_size)  % add a chunk if necessary
                    newsize = obj.curr_size + obj.chunk_size;
                    tmp = zeros(obj.data_size,newsize);
                    tmp(:,1:obj.curr_size) = obj.data(:,1:obj.curr_size);
                    obj.data = tmp;                   
                end
                obj.data(:,obj.N) = value;
            end
        end
        
        function out = value(obj,key)
            try
                index   = obj.hash(key);
                out     = obj.data(:,index);
            catch
                error(['myMap.values: key = ' key ' not found '])
            end
        end
       
    end
end


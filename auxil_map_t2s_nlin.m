function imgmap = auxil_map_t2s_nlin(imgstack, te, msk)
   
    [xres, yres, zres, ~] = size(imgstack);
    
    imgmap = zeros(xres, yres, zres);
    
    fopts = fitoptions(...
        'Method', 'NonlinearLeastSquares',...
        'Lower', [0, 1],...
        'Upper', [5, 200],...
        'Startpoint', [1.0, 30],...
        'DiffMinChange', 1.0e-4,...
        'DiffMaxChange', 0.1,...
        'MaxIter', 100);
    
    ftype = fittype(...
        'm0 * exp(-TEarr./T2s)',...
        'dependent', {'s'},...
        'independent', {'TEarr'},...
        'coefficients', {'m0', 'T2s'});
    
    for xidx = 1: xres
        for yidx = 1: yres
            for zidx = 1: zres
                if msk(xidx, yidx, zidx) > 0
                    
                    sig = imgstack(xidx, yidx, zidx, :);
                    sig = sig(:) / sig(1);
                    
                    fobj = fit(te(:), sig(:), ftype, fopts);
                    
                    t2s = fobj.T2s;
                    
                    imgmap(xidx, yidx, zidx) = t2s;
                end
            end
        end
    end

end
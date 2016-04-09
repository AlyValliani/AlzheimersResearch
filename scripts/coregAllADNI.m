function coregAllADNI()
    
    inputs = cell(0, 5000); %throwaway input
    
    spm_jobman('initcfg');
    
    paths = {'/sonigroup/bl_ADNI/AD/',
             '/sonigroup/bl_ADNI/CN/',
             '/sonigroup/bl_ADNI/LMCI/',
             '/sonigroup/bl_ADNI/MCI/'};
    
    for path_i = 1:numel(paths)
        path = paths{path_i};
        
        listing = dir(path);
        
        for listing_i = 1:numel(listing)
            if listing(listing_i).isdir || ~strcmp(listing(listing_i).name(end-3:end), ...
                                           '.nii')
                continue
            end
            filename = strcat(path, listing(listing_i).name);
            theJob = coregistration(filename);
            spm('defaults', 'FMRI');
            spm_jobman('serial', theJob, '', inputs{:});
        end
    end
    
end

function matlabbatch = coregistration(filename)
    fprintf('Current file is %s\n',filename);
    matlabbatch{1}.spm.spatial.coreg.estwrite.ref = {'/sonigroup/bl_ADNI/template.nii,1'};
    matlabbatch{1}.spm.spatial.coreg.estwrite.source = {[filename,',1']};
    matlabbatch{1}.spm.spatial.coreg.estwrite.other = {''};
    matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
    matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];
    matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
    matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];
    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.interp = 1;
    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];
    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.mask = 0;
    matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';
end

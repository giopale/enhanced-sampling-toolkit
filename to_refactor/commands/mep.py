from rich import print
from scipy.interpolate import griddata
import sys, os
import numpy as np
from logger_setup import logger
from pathlib import Path
from omegaconf import ListConfig
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from ectoolkits_reweight import read_fes
from stringmethod_custom import String2D

from utils import find_job_dirs, find_files, join_if_list, config_to_list, extract_field_map
from utils import compute_mep, save_mep, load_mep, apply_mpl_style
import matplotlib
matplotlib.use('Agg')




def run(cfg,files=None):
    print(f"[bold green]Running program mep:[/bold green] {cfg.param}")

    
    apply_mpl_style(cfg.mplstyle.dir,cfg.mplstyle.style)
    just_average=cfg.get('just_average',False)
    cvs_selected=cfg.cvs.get('select',None)


    assert cvs_selected is not None
   
    cvs = cvs_selected 

    files_provide=False
    if files is None:
        # find job dirs and files
        jobdirs=find_job_dirs(cfg.dir_md, cfg.dir_job)
        files=[]
        # make a list of cvs
        for jj in jobdirs:
            # find the cvs to look for
            cvs=extract_field_map(jj/'COLVAR').keys()
            cvs=[f for f in cvs if f != 'time' and not f.startswith('opes.')]
            msg=''
            if cvs_selected is not None:
                msg=' from config'
                cvs=cvs_selected
            logger.info(f'selected cvs{msg}: {cvs}')
        # make a list of reweighted fes
        for jj in jobdirs:
            ff=find_files(jj, f'pot*-{cvs[0]}-{cvs[1]}-rew-fes.dat')
            files+=ff
    
        # add the file with the average fes to the list
        ave_present=False
        file_ave=find_files(Path(cfg.dir_md), f'{cvs[0]}-{cvs[1]}-rew-fes-AVG.dat')
        if file_ave is not None:
            files+=file_ave
            # discard single potential files if requested
            if just_average:
                files=file_ave
    else:
        files_provided=True

    # flag the presence of the average fes file 
    if any('AVG' in s.name for s in files):
        ave_present=True

    if len(files) < 1:
        ff=find_files('.',cfg.file_fes)
        files+=ff

    # exit if no files found
    if len(files) < 1:
        logger.info('no files found. Exiting')
        sys.exit()

    if not all(Path(f).exists() for f in files):
        sys.exit('not all files are there. exiting')

    # get property of interest
    prop=cfg.get('property', None)
    if prop is None:
        print('No poperty to analyze in config file. Exiting.')
        sys.exit()

    # extract configurations
    ###################################
    logger.info(f'plotting MEP for {prop}')
    cfg_prop=cfg.get('surface_data',{}).get(prop,{})
    pp=cfg_prop.plot
    mm=cfg_prop.mep

    ppm=mm.plot
    ###################################

    # plot settings
    row=pp.get('row',1)
    col=pp.get('col',1)
    if just_average:
        row, col = 1,1
    elif len(files)==1 and not 'AVG' in files[0].name:
        row, col = 1,1
    else:
        col=2
        row=len(files)//2 +1
    scale=ppm.get('scale',1)
    size=ppm.get('size',[4,3.5])
    hspace=ppm.get('hspace', 0.3)
    fontsize=ppm.get('fontsize',6)
    plt.rcParams.update({
        "axes.labelsize": fontsize,     # x/y/z axis labels
        "xtick.labelsize": .8*fontsize,    # x tick labels
        "ytick.labelsize": .8*fontsize,    # y tick labels
    })
   
    # initialize figure and subplots
    fig = plt.figure(figsize=(size[0]*scale*col, size[1]*scale*row), dpi=400)
    fig.subplots_adjust(hspace=hspace)
    gs = fig.add_gridspec(row,col) 

    # init arrays
    mep_list = []
    mep_max=np.zeros(mm.npts)
    mep_min=np.zeros(mm.npts)
    idx_ave=-1
    meps=[]
   
    dir_base_out=Path('.')
    # cycle through files
    for ff in files:
        if not 'AVG' in ff.name:
            dir_base_out=ff.parent

        # skip non-average fes if requested
        if not mm.get('compute_all',False):
            if not 'AVG' in ff.name:
                continue
    
        # select or init the target file name for mep
        fname_mep=mm.get('fname_mep',None)
        if fname_mep is None:
            fname_mep=f'{cvs[0]}-{cvs[1]}-{prop}-MEP.dat'
        if not '/' in str(fname_mep):
            fname_mep=dir_base_out/fname_mep

        fname_mep_out=mm.get('fname_mep_out',None)
        if fname_mep_out is None:
            fname_mep_out = fname_mep
        
        if  mm.load:
            # read mep points, then evaluate energy over fes loaded from ff
            mep_fes=load_mep(fname_mep) # load everything, then just the coordinates are used for plotting
            X, Y, Z, _, _ = read_fes(ff, dim=2)
            Z=Z.clip(pp.vmin, pp.vmax)

            mep_fes_x = mep_fes[:,0]
            mep_fes_y = mep_fes[:,1]

            grid = np.vstack([X.ravel(), Y.ravel()]).T
            # evaluate
            prop_mep = griddata(grid, Z.ravel(), np.array([mep_fes_x, mep_fes_y]).T, method='linear')
            mep_fes[:,2]=prop_mep
            # write to file
            save_mep(mep_fes,fname_mep_out)
        else:
            # load fes
            logger.info(f' >>>>>>> computing mep for {str(ff)}')
            X, Y, Z, _, _ = read_fes(ff, dim=2)
            Z=Z.clip(pp.vmin, pp.vmax)
            # optimize mep
            mep_fes=compute_mep(X,Y,Z,**mm)
            # save to file
            save_mep(mep_fes,fname_mep_out)
        # make a list of results
        meps.append(mep_fes[:,:3])

    # identify where in the list is the mep on the average surface    
    analyze_average=mm.get('analyze_average',True)
    index_ave= next((i for i, s in enumerate([ff.name for ff in files]) if 'AVG' in s), None) if analyze_average else None
    has_average = True if index_ave is not None else False
    has_average = has_average and analyze_average

    mep_ave=None
    ave_meps_jobs=[]
    meps_dict={}

    # cycle through files
    for idx, ff in enumerate(files):

        if index_ave is not None:
            # select the average mep if present
            mep_fes=meps[index_ave]
        else:
            # otherwise take mep from each potential
            mep_fes=meps[idx]
    
        # load fes
        X, Y, Z, _, _ = read_fes(ff, dim=2)
        Z=Z.clip(pp.vmin, pp.vmax)
        
        # load path points 
        mep_fes_x = mep_fes[:,0]
        mep_fes_y = mep_fes[:,1]
        
        # evaluate energy at the selected points
        # here the fes is the one coming from each different file
        grid = np.vstack([X.ravel(), Y.ravel()]).T
        prop_mep = griddata(grid, Z.ravel(), np.array([mep_fes_x, mep_fes_y]).T, method='linear')

        # populate a list to then make an average
        if idx != index_ave:
            ave_meps_jobs.append(prop_mep)

        # populate a dictionary to then make an average
        is_job_file = ('job-0' in ff.parent.name) or ('pot0' in ff.name)
        #if 'job-0' in ff.parent.name:
        if is_job_file:
            if 'job-0' in ff.parent.name:
                number_job=int(str(ff.parent.name).split('-')[1])
            elif 'pot' in ff.name:
                number_job=int(str(ff.name).split('-')[0][3:])
            else:
                logger.info('could not determine job number for file {ff.name}')
                sys.exit()
            meps_dict.update({f'job-{number_job:03d}':prop_mep})

        # add axis to the figure
        ax  = fig.add_subplot(gs[idx])

        # select green for average
        if 'AVG' in ff.name:
            color='tab:green'
        else:
            color='tab:orange'
        # plot the mep
        ax.plot(np.linspace(0, 1, len(prop_mep)), prop_mep, color=color)

        # set the decorations to the axis
        xlabel=ppm.get('xlabel','reaction coordinate')
        ax.set_xlabel(xlabel)

        ylabel=ppm.get('ylabel','energy (eV)')
        ax.set_ylabel(ylabel)
        if 'AVG' in ff.name:
            ax.set_title(f"MEP avg fes")
            # save an alias for later
            ax_ave=ax
        else:
            #if 'job-00' in ff.parent.name:
            if is_job_file:
                #jobnum=int(ff.parent.name.split('-')[-1])
                jobnum=number_job
            else:
                jobnum=idx
            ax.set_title(f"MEP-MLP-{jobnum}")
            mep_list.append(prop_mep)
        # save the average mep
        if 'AVG' in ff.name:
            mep_to_save=mep_fes
            fname_mep=f'{cvs[0]}-{cvs[1]}-{prop}-AVG-MEP.dat'
        
    # calculate max, min, mean, std of the meps
    mep_max=np.max(ave_meps_jobs,axis=0)
    mep_min=np.min(ave_meps_jobs,axis=0)
#    #mep_ave=np.mean(ave_meps_jobs,axis=0)
#    if has_average:
#        mep_ave=meps[index_ave][:,2]
#        mep_std=np.std(ave_meps_jobs,axis=0)
#
#        # make an array of values to save
#        more_columns=np.vstack((mep_min,mep_max,mep_std,meps_dict['job-000'],meps_dict['job-001'],meps_dict['job-002'],meps_dict['job-003'])).T
#        # re-save average mep including deviations
#        header=f'{cvs[0]}, {cvs[1]}, energy, min, max, std, job-000, job-001, job-002, job-003,'
#        #print('mep_to_save, more_columns',np.shape(mep_to_save), np.shape(more_columns))
#        save_mep(mep_to_save,fname_mep,header=header,more_columns=more_columns)
#        
#        # fill the interval between minimum and maximum mep 
#        ax_ave.fill_between(np.linspace(0, 1, len(prop_mep)), mep_min, mep_max, color=color, alpha=0.3)    
    if has_average:
        mep_ave = meps[index_ave][:, 2]
        mep_std = np.std(ave_meps_jobs, axis=0)
    
        # Collect whichever jobs are present, up to job-015
        job_columns = []
        job_labels = []
    
        for i in range(16):
            job = f'job-{i:03d}'
            if job in meps_dict:
                logger.info(f'{job} found')
                job_columns.append(meps_dict[job])
                job_labels.append(job)
    
        # Make an array of values to save
        more_columns = np.vstack(
            (mep_min, mep_max, mep_std, *job_columns)
        ).T
    
        # Re-save average MEP including deviations
        header = (
            f'{cvs[0]}, {cvs[1]}, energy, min, max, std, '
            + ', '.join(job_labels)
            + ','
        )
    
        save_mep(
            mep_to_save,
            fname_mep,
            header=header,
            more_columns=more_columns,
        )
    
        # Fill the interval between minimum and maximum MEP
        ax_ave.fill_between(
            np.linspace(0, 1, len(prop_mep)),
            mep_min,
            mep_max,
            color=color,
            alpha=0.3,
        )


    # layout adjustment
    adjust=ppm.get('subplots_adjust',None)
    if adjust is not None:
        plt.subplots_adjust(**adjust)
    else:
        fig.tight_layout()

    # plot file name
    formats=cfg_prop.get('save',{}).get('plot',['.png'])
    for ff in formats:
        fnew=f"{cvs[0]}-{cvs[1]}-mep"
        if just_average:
            fnew+='AVG'
        fnew += ff
        fname_plot=Path(os.getcwd())/fnew
        # write plot to file
        logger.info(f'saving file {fname_plot}')
        fig.savefig(fname_plot, dpi=300)

    plt.close()

    if has_average:
        # make a plot with the mean
        mep_list = np.array(mep_list)
        
        row = 1
        col = 1
        fig = plt.figure(figsize=(size[0]*scale*col, size[1]*scale*row), dpi=400)
        gs = fig.add_gridspec(row,col)
        ax  = fig.add_subplot(gs[0])
        react_coord = np.linspace(0, 1, mep_list.shape[1])
        # plot the mean of the paths
        mep_std = mep_list.std(axis=0)
        argmax_mep_ave = mep_ave.argmax()
        color_avg = ppm.get('colors',{}).get('avg','tab:green') 
        ax.plot(react_coord, mep_ave, color=color_avg)
        ax.plot(react_coord[0], mep_ave[0], marker='o',markerfacecolor='green',markeredgecolor='y',)
        ax.plot(react_coord[-1], mep_ave[-1], marker='o',markerfacecolor='red',markeredgecolor='y')
        ax.fill_between(react_coord, mep_ave-mep_std, mep_ave+mep_std, alpha=0.3, color=color_avg, ec=None)

        xlabel=ppm.get('xlabel','reaction coordinate')
        ylabel=ppm.get('ylabel','energy (eV)')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.text(0.3, 0.2, '$\Delta F^{\ddag}$='+f"{mep_ave[argmax_mep_ave]:.2f}"+r"$\pm$"+f"{mep_std[argmax_mep_ave]:.2f} eV", transform=ax.transAxes, va='bottom')
        ax.text(0.3, 0.2, '$\Delta F$='f"{mep_ave[-1]-mep_ave[0]:.2f}"+r"$\pm$"+f"{np.sqrt(mep_std[0]**2 +mep_std[-1]**2):.2f} eV", transform=ax.transAxes,va='top')
        limits = np.array(ppm.get('limits',[None,None,None,None]))
        ax.set_xlim(limits[:2])
        ax.set_ylim(limits[2:])

        if adjust is not None:
            plt.subplots_adjust(**adjust)
        else:
            fig.tight_layout()
        #ax.legend()
        
#        save average plot
        #for ff in config_to_list(mm.save.plot):
        for ff in formats:
            tmpname=f'{cvs[0]}-{cvs[1]}-AVG-MEP'+ff
            fname_plot=Path(os.getcwd())/tmpname
            logger.info(f'saving file {fname_plot}')
            fig.savefig(fname_plot, dpi=300)














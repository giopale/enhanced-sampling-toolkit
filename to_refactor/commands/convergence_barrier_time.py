from rich import print
import shutil
import sys, os
import numpy as np
from logger_setup import logger
from pathlib import Path
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from utils import apply_mpl_style, find_job_dirs

from commands.reweight_property import run as reweight_property
from commands.mep import run as mep


def run(cfg):
    print(f"[bold green]Running program {cfg._target_.split('.')[1]}:[/bold green] {cfg.param}")

    apply_mpl_style(cfg.mplstyle.dir,cfg.mplstyle.style)

    analyze_average=cfg.get('convergence_barrier',{}).get('analyze_average',False)
    cc=cfg.get('convergence_barrier',{}).get('time',{})
    cvs=cfg.get('cvs').get('select',None)
    assert cvs is not None
    figsize=[ii for ii in cfg.surface_data.fes.plot.size]

    time_interval=list(cc.get('time_interval',None))

    if not (
        isinstance(time_interval, (list, tuple, np.ndarray))
        and len(time_interval) == 2
        and all(isinstance(v, (int, float, np.integer, np.floating)) for v in time_interval)
    ):
        logger.error("You must provide exactly two numbers via convergence_barrier.time_interval")
        logger.error(f"now: {time_interval}")
        return


    time_step=cc.get('time_step',0.5)
        
    # select an array of times
    times=np.arange(time_interval[0]+time_step,time_interval[1]+time_step/2,time_step)
    
    # make the labels
    labels=np.array([f'{1000*tt:010.0f}' for tt in times],dtype=str)
    
    # make sure not to load mep
    cfg.surface_data.fes.mep.load=False
    
    file_mep_last=None

    meps=[]


    jobdirs = find_job_dirs(Path(cfg.get('dir_md', '.')), cfg.dir_job)

    if not analyze_average:
        if len(jobdirs) >1:
            logger.error(f'This program works on one dir_job at a time.\nNow selected: {[str(ii) for ii in jobdirs]}')
            sys.exit()
        jj=Path(jobdirs[0])
    else:
        jj=Path(cfg.get('dir_md','.'))

    # set file labels
    ll_avg='-AVG' if analyze_average else ''
    ll_pot='' if analyze_average else f'pot{jj}-' 

    dirdata=jj/'convergence_barrier'
    if not dirdata.is_dir():
        logger.error(f'you must manually create the directory {dirdata}')
        sys.exit()

    fname_mep=cc.get('fname_mep',None)
    if fname_mep is None:
        fname_mep=f'{cvs[0]}-{cvs[1]}-fes{ll_avg}-MEP.dat'

    fname_mep=dirdata/fname_mep

    if not fname_mep.is_file():
        logger.error(f'you must provide the MEP file: {fname_mep}')
        sys.exit()
    logger.info(f'MEP file found: {fname_mep}')



    for index, ii in enumerate(labels[::-1]):
        logger.info(f'--------------------- BEGIN STEP {index+1}/{len(labels)} ---------------------------------------')
        logger.info('#'*100+f' reweighting 0 -> {int(ii)/1000:.2f} ns')
        dirtime=dirdata/f'{ii}'
        dirtime.mkdir(parents=True, exist_ok=True)

        cfg.time_interval=[0,float(ii)/1000]
        
        ## REWEIGHT RUN for current time interval
        do_reweight=cc.get('do_reweight',True)
        if do_reweight:
            reweight_property(cfg)
    
        logger.info(f'--------------------- END OF REWEIGHT {index+1}/{len(labels)} ---------------------------------------')

        files_done=[] # store the reweighted fes-s for the current step
        file_pot=f"{jj}/{ll_pot}{cvs[0]}-{cvs[1]}-rew-fes{ll_avg}.dat"
        file_pot_time=f"{ll_pot}{cvs[0]}-{cvs[1]}-rew-fes{ll_avg}-{ii}.dat"
        if do_reweight:
            shutil.copy2(file_pot,dirtime/file_pot_time)
        files_done.append(dirtime/file_pot_time)


        # store average potential mep file
        #file_pot_ave=f"{cvs[0]}-{cvs[1]}-rew-fes-AVG.dat"
        #file_pot_ave_time=f"{cvs[0]}-{cvs[1]}-rew-fes-AVG-{ii}.dat"
        #if do_reweight:
        #    shutil.copy2(file_pot_ave,dirtime/file_pot_ave_time)

        # list the reweighted fes-ave of the current step
        #files_done.append(dirtime/file_pot_ave_time)
        
        # print copied files
        done="\n".join([s.name for s in files_done])
        logger.info(f'files copied: {done}')

        # configure mep run
        fname_mep_time=dirtime/f'{cvs[0]}-{cvs[1]}-rew-fes-{ii}-mep.dat'
       # if ii != labels[-1]:
       #     # load mep of the last time available (labels[-1])
       #     cfg.surface_data.fes.mep.fname_mep=file_mep_last
       #     # save the result as the result of the current step (ii)
       #     cfg.surface_data.fes.mep.fname_mep_out=dirtime/f'{cvs[0]}-{cvs[1]}-rew-fes-AVG-{ii}-mep.dat'
       #     cfg.surface_data.fes.mep.load=True
        #elif fmep_external is not None:
        cfg.surface_data.fes.mep.fname_mep=fname_mep
        cfg.surface_data.fes.mep.fname_mep_out=fname_mep_time
        cfg.surface_data.fes.mep.load=True
        cfg.surface_data.fes.mep.analyze_average=False
        #else:
        #    # optimize mep
        #    print('fmep_external',fmep_external,'fname_mep',cfg.surface_data.fes.mep.fname_mep)
        #    cfg.surface_data.fes.mep.fname_mep=None
        #    cfg.surface_data.fes.mep.fname_mep_out=fmep

        #### MEP RUN ### 
        mep(cfg,files=files_done)

        file_mep=f"{cvs[0]}-{cvs[1]}-mep-{ii}.png"
        #file_mep_ave=f"{cvs[0]}-{cvs[1]}-AVG-MEP-{ii}.png"
        shutil.copy2(f"{cvs[0]}-{cvs[1]}-mep.png",dirtime/file_mep)
        #shutil.copy2(f"{cvs[0]}-{cvs[1]}-AVG-MEP.png",dirtime/file_mep_ave)
        if ii == labels[-1]:
            file_mep_last=fname_mep_time

        #meps.append(np.loadtxt(fmep)[:,:3])

        done="\n".join([str(s) for s in [dirtime/file_mep]])
        logger.info(f'files copied: {done}')
        logger.info(f'--------------------- END OF STEP {index+1}/{len(labels)} ---------------------------------------')

    for ii in labels[::-1]:
        dirtime=dirdata/f'{ii}'
        fmep=dirtime/f'{cvs[0]}-{cvs[1]}-rew-fes-{ii}-mep.dat'
        meps.append(np.loadtxt(fmep)[:,:3])
        print('I am loading',fmep,np.shape(np.loadtxt(fmep)[:,:3]))
        

    meps=np.array(meps)[::-1]
    barriers=[np.max(ii) - np.min(ii) for ii in meps[:,:,2]]
    react_ener=[ii[-1] - ii[0] for ii in meps[:,:,2]]
    np.savetxt(dirdata/'convergence_barrier.dat',np.array([times, barriers,react_ener]).T)
    print(times)
    print(barriers)
    print(react_ener)

    fig, ax=plt.subplots(figsize=figsize)    
    ax.plot(times,barriers, ls='-', marker='',label='$\Delta F^{\ddag}:$'+f' ${barriers[-1]:.2f}$ eV @ {times[-1]:.1f} ns',color='tab:blue',lw=.7)
    ax.plot(times,react_ener, ls='-', marker='',label='$\Delta F:$' +f' ${react_ener[-1]:.2f}$ eV @ {times[-1]:.1f} ns',color='tab:green',lw=.7)
    ax.axhline(0.0,ls='-',color='k',lw=.3)
    ax.set_xlabel('simulation time (ns)')
    ax.set_ylabel('energy (eV)')
    ax.set(xlim=[0,None],ylim=[None,None])

    plt.legend()
    fname_plot=dirdata/'convergence_barrier.png'
    plt.tight_layout()
    plt.savefig(fname_plot)
    logger.info(f'file saved {fname_plot}')
    




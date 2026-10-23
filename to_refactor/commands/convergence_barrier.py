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

    cct=cfg.get('convergence_barrier',{}).get('time',{})
    cvs=cfg.get('cvs').get('select',None)
    assert cvs is not None

    time_interval=list(cct.get('time_interval',None))

    if not (
        isinstance(time_interval, (list, tuple, np.ndarray))
        and len(time_interval) == 2
        and all(isinstance(v, (int, float, np.integer, np.floating)) for v in time_interval)
    ):
        logger.error("You must provide exactly two numbers via convergence_barrier.time_interval")
        logger.error(f"now: {time_interval}")
        return


    time_step=cct.get('time_step',0.5)
        
    # select an array of times
    times=np.arange(time_interval[0]+time_step,time_interval[1]+time_step/2,time_step)
    
    # make the labels
    labels=np.array([f'{1000*tt:010.0f}' for tt in times],dtype=str)
    
    dirdata=Path('convergence_barrier')
    # make sure not to load mep
    cfg.surface_data.fes.mep.load=False
    
    file_mep_last=None

    meps=[]

    fmep_external=cfg.get('convergence_barrier',{}).get('fmep_external',None)

    jobdirs = find_job_dirs(Path(cfg.get('dir_md', '.')), cfg.dir_job)


    for index, ii in enumerate(labels[::-1]):
        logger.info(f'--------------------- BEGIN STEP {index+1}/{len(labels)} ---------------------------------------')
        logger.info('#'*100+f' reweighting 0 -> {int(ii)/1000:.2f} ns')
        dirtime=dirdata/f'{ii}'
        dirtime.mkdir(parents=True, exist_ok=True)

        cfg.time_interval=[0,float(ii)/1000]
        
        # do reweight for the current time interval
        do_reweight=cfg.get('convergence_barrier',{}).get('do_reweight',True)
        if do_reweight:
            reweight_property(cfg)
    
        logger.info(f'--------------------- END OF REWEIGHT {index+1}/{len(labels)} ---------------------------------------')

        files_done=[] # store the reweighted fes-s for the current step
        for jj in jobdirs:
            file_pot=f"{jj}/pot{jj}-{cvs[0]}-{cvs[1]}-rew-fes.dat"
            file_pot_time=f"pot{jj}-{cvs[0]}-{cvs[1]}-rew-fes-{ii}.dat"
            if do_reweight:
                shutil.copy2(file_pot,dirtime/file_pot_time)
            files_done.append(dirtime/file_pot_time)

        # store average potential mep file
        file_pot_ave=f"{cvs[0]}-{cvs[1]}-rew-fes-AVG.dat"
        file_pot_ave_time=f"{cvs[0]}-{cvs[1]}-rew-fes-AVG-{ii}.dat"
        if do_reweight:
            shutil.copy2(file_pot_ave,dirtime/file_pot_ave_time)

        # list the reweighted fes-ave of the current step
        files_done.append(dirtime/file_pot_ave_time)
        
        # print copied files
        done="\n".join([s.name for s in files_done])
        logger.info(f'files copied: {done}')

        # configure mep run
        fmep=dirtime/f'{cvs[0]}-{cvs[1]}-rew-fes-AVG-{ii}-mep.dat'
        if ii != labels[-1]:
            # load mep of the last time available (labels[-1])
            cfg.surface_data.fes.mep.fname_mep=file_mep_last
            # save the result as the result of the current step (ii)
            cfg.surface_data.fes.mep.fname_mep_out=dirtime/f'{cvs[0]}-{cvs[1]}-rew-fes-AVG-{ii}-mep.dat'
            cfg.surface_data.fes.mep.load=True
        elif fmep_external is not None:
            cfg.surface_data.fes.mep.fname_mep=Path(fmep_external)
            cfg.surface_data.fes.mep.fname_mep_out=dirtime/f'fes-{cvs[0]}-{cvs[1]}-rew-fes-AVG-{ii}-mep.dat'
            cfg.surface_data.fes.mep.load=True
        else:
            # optimize mep
            print('fmep_external',fmep_external,'fname_mep',cfg.surface_data.fes.mep.fname_mep)
            cfg.surface_data.fes.mep.fname_mep=None
            cfg.surface_data.fes.mep.fname_mep_out=fmep

        #### MEP RUN ### 
        mep(cfg,files=files_done)

        file_mep=f"{cvs[0]}-{cvs[1]}-mep-{ii}.png"
        file_mep_ave=f"{cvs[0]}-{cvs[1]}-AVG-MEP-{ii}.png"
        shutil.copy2(f"{cvs[0]}-{cvs[1]}-mep.png",dirtime/file_mep)
        shutil.copy2(f"{cvs[0]}-{cvs[1]}-AVG-MEP.png",dirtime/file_mep_ave)
        if ii == labels[-1]:
            file_mep_last=fmep

        #meps.append(np.loadtxt(fmep)[:,:3])

        done="\n".join([str(s) for s in [dirtime/file_mep,dirtime/file_mep_ave]])
        logger.info(f'files copied: {done}')
        logger.info(f'--------------------- END OF STEP {index+1}/{len(labels)} ---------------------------------------')

    for ii in labels[::-1]:
        dirtime=dirdata/f'{ii}'
        fmep=dirtime/f'{cvs[0]}-{cvs[1]}-rew-fes-AVG-{ii}-mep.dat'
        meps.append(np.loadtxt(fmep)[:,:3])
        print('I am loading',fmep,np.shape(np.loadtxt(fmep)[:,:3]))
        

    meps=np.array(meps)[::-1]
    barriers=[np.max(ii) - np.min(ii) for ii in meps[:,:,2]]
    react_ener=[ii[-1] - ii[0] for ii in meps[:,:,2]]
    np.savetxt(dirdata/'convergence-barrier.dat',np.array([times, barriers,react_ener]).T)
    print(times)
    print(barriers)
    print(react_ener)

    fig, ax=plt.subplots()    
    ax.plot(times,barriers, ls='-', marker='.',label='$\Delta F^{\ddag}$',color='tab:blue')
    ax.plot(times,react_ener, ls='-', marker='.',label='$\Delta F$',color='tab:green')
    ax.set_xlabel('simulation time (ns)')
    ax.set_ylabel('energy (eV)')
    ax.set(xlim=[0,None],ylim=[0,None])

    plt.legend()
    fname_plot='convergence-barrier.png'
    plt.tight_layout()
    plt.savefig(fname_plot)
    logger.info(f'file saved {fname_plot}')
    




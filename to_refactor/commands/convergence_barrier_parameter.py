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

    cc_parameter=cfg.get('convergence_barrier',{}).get('parameter',{})
    param = cc_parameter.get('select',None)
    if '_' in param:
        name_param, cv_param=param.split('_')
    if param is None:
        logger.error(f'You must select one parameter via convergence_barrier.parameter.select. Now: {param}')
    cc=cc_parameter.get(param,{})

    cvs=cfg.get('cvs').get('select',None)
    assert cvs is not None

    funct_mesh_name=[ii for ii in cc.get('funct_mesh','linspace').keys()][0]

    kwargs_mesh=cc.get('funct_mesh',{}).get(funct_mesh_name,{'start':0.1,'stop':1,'num':10})

    txt=' '.join([f'{key}:{val}' for key, val in kwargs_mesh.items()])
    logger.info(f'spacing through np.{funct_mesh_name} using {txt}')

    # select a function to make a mesh of points
    funct_mesh=getattr(np,funct_mesh_name)
    # make the mesh
    mesh=funct_mesh(**kwargs_mesh)

    logger.info(f'running test on parameter {param}')
    txt=[f'{ii:.3g}' for ii in mesh]
    logger.info(f'calculating {len(mesh)} points: {" ".join(txt)}')

    labels = [
        f"d{-np.log10(value):012.4f}"
        for value in mesh
    ] 

    meps=[]


    jobdirs = find_job_dirs(Path(cfg.get('dir_md', '.')), cfg.dir_job)

    if len(jobdirs) >1:
        logger.error(f'This program works on one dir_job at a time.\nNow selected: {[str(ii) for ii in jobdirs]}')
        sys.exit()
    jj=Path(jobdirs[0])

    dirdata=jj/f'convergence_barrier_{param}'
    if not dirdata.is_dir():
        logger.error(f'you must manually create the directory {dirdata}')
        sys.exit()

    fname_mep=cc.get('fname_mep',None)
    if fname_mep is None:
        fname_mep=f'{cvs[0]}-{cvs[1]}-fes-MEP.dat'

    fname_mep=dirdata/fname_mep

    if not fname_mep.is_file():
        logger.error(f'you must provide the MEP file: {fname_mep}')
        sys.exit()
    logger.info(f'MEP file found: {fname_mep}')


    


    for ii, (name, value) in enumerate(zip(labels[::-1], mesh[::-1])):
        logger.info(f'--------------------- BEGIN STEP {ii+1}/{len(mesh)} ---------------------------------------')
        logger.info('#'*100+f' reweighting {param}:{value:.4g}')
        dirtime=dirdata/name
        dirtime.mkdir(parents=True, exist_ok=True)

        # set the parameter
        cfg.cvs[cv_param][name_param]=float(value)

        ## REWEIGHT RUN for current time interval
        do_reweight=cc.get('do_reweight',True)
        if do_reweight:
            reweight_property(cfg)
    
        logger.info(f'--------------------- END OF REWEIGHT {ii+1} ---------------------------------------')

        files_done=[] # store the reweighted fes-s for the current step
        file_pot=f"{jj}/pot{jj}-{cvs[0]}-{cvs[1]}-rew-fes.dat"
        file_pot_param=f"pot{jj}-{cvs[0]}-{cvs[1]}-rew-fes-{name}.dat"
        if do_reweight:
            shutil.copy2(file_pot,dirtime/file_pot_param)
        files_done.append(dirtime/file_pot_param)

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
        fname_mep_time=dirtime/f'{cvs[0]}-{cvs[1]}-rew-fes-{name}-mep.dat'
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
        #else:
        #    # optimize mep
        #    print('fmep_external',fmep_external,'fname_mep',cfg.surface_data.fes.mep.fname_mep)
        #    cfg.surface_data.fes.mep.fname_mep=None
        #    cfg.surface_data.fes.mep.fname_mep_out=fmep

        #### MEP RUN ### 
        mep(cfg,files=files_done)

        file_mep=f"{cvs[0]}-{cvs[1]}-mep-{name}.png"
        #file_mep_ave=f"{cvs[0]}-{cvs[1]}-AVG-MEP-{ii}.png"
        shutil.copy2(f"{cvs[0]}-{cvs[1]}-mep.png",dirtime/file_mep)
        #shutil.copy2(f"{cvs[0]}-{cvs[1]}-AVG-MEP.png",dirtime/file_mep_ave)
        if name == labels[-1]:
            file_mep_last=fname_mep_time

        #meps.append(np.loadtxt(fmep)[:,:3])

        done="\n".join([str(s) for s in [dirtime/file_mep]])
        logger.info(f'files copied: {done}')
        logger.info(f'--------------------- END OF STEP {ii+1}/{len(mesh)} ---------------------------------------')

    for name in labels[::-1]:
        dirtime=dirdata/f'{name}'
        fmep=dirtime/f'{cvs[0]}-{cvs[1]}-rew-fes-{name}-mep.dat'
        meps.append(np.loadtxt(fmep)[:,:3])
        print('I am loading',fmep,np.shape(np.loadtxt(fmep)[:,:3]))
        

    meps=np.array(meps)[::-1]
    barriers=[np.max(ii) - np.min(ii) for ii in meps[:,:,2]]
    react_ener=[ii[-1] - ii[0] for ii in meps[:,:,2]]
    np.savetxt(dirdata/'convergence_barrier_{param}.dat',np.array([mesh, barriers,react_ener]).T)
    print(mesh)
    print(barriers)
    print('differences:',np.diff(barriers))
    print(react_ener)

    fig, ax=plt.subplots(figsize=(3,3*3/4))    
    idx_select=0
    ax.plot(mesh,barriers, ls='-', marker='',label='$\Delta F^{\ddag}:$'+f' ${barriers[idx_select]:.2f}$ eV @ {param}={mesh[idx_select]:.4g}',color='tab:blue',lw=.7)
    ax.plot(mesh,react_ener, ls='-', marker='',label='$\Delta F:$' +f' ${react_ener[idx_select]:.2f}$ eV @ {param}={mesh[idx_select]:.4g}',color='tab:green',lw=.7)
    ax.set_xscale('log')
    ax.axhline(0.0,ls='-',color='k',lw=.3)
    ax.set_xlabel(cc.get('xlabel',f'{param}'))
    ax.set_ylabel('energy (eV)')
    ax.set(xlim=[0,None],ylim=[None,None])

    plt.legend()
    fname_plot=dirdata/f'convergence_barrier_{param}.png'
    plt.tight_layout()
    plt.savefig(fname_plot)
    logger.info(f'file saved {fname_plot}')
    




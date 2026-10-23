from rich import print
import sys, os
import numpy as np
from logger_setup import logger
from pathlib import Path
from omegaconf import ListConfig
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from utils import find_job_dirs, find_files, join_if_list
plt.style.use('tableau-colorblind10')


def run(cfg):
    print(f"[bold green]Running program {cfg._target_.split('.')[1]}:[/bold green] {cfg.param}")



    # extract relevant cfg
    logger.info(f'plotting model deviation')

    # find joob dirs and files
    jobdirs=find_job_dirs(cfg.dir_md, cfg.dir_job)
    files=[]
    for jj in jobdirs:
        ff=find_files(jj, cfg.file_model_devi)
        files+=ff

    # plot settings
    if len(files)==1:
        row, col = 1,1
    else:
        row, col = 2,2



    scale=cfg.get('scale',1)
    #size=cfg.get('size',[4,3.5])
    #hspace=cfg.get('hspace', 0.3)
    width=4
    aspect=1
    size=[width,width/aspect]
   
    fig = plt.figure(figsize=(size[0]*scale*col, size[1]*scale*row), dpi=400)
    gs = fig.add_gridspec(2, 5, width_ratios=[4, 1,0.8, 4, 1],wspace=0) 

    color='tab:pink'
    color=cfg.get('color','tab:pink')


    times=[]
    datas=[]
    maxx=-1
    for idx, ff in enumerate(files):

        if 'job-00' in ff.parent.name:
            jobnum=int(ff.parent.name.split('-')[-1])
        else:
            jobnum=idx

        # data
        data=np.loadtxt(ff)
        tstep=0.0005
        logger.info(f'hardcoded TIMESTEP {tstep} ps')
        time=data[:,0]*tstep/1000
        with open(ff, 'r') as f:
            header=f.readline()

        fields=header.split()[1:]
        logger.info(f'file {ff} has FIELDS {" ".join(fields)}')
        fplot='max_devi_f'
        logger.info(f'plotting field {fplot}')
        field_map = {field: idx  for idx, field in enumerate(fields)}
        
        times.append(time)
        datas.append(data[:,field_map[fplot]])
        lower, upper = np.percentile(data[:,field_map[fplot]], [0, 99])
        maxx=np.max((maxx,upper))

    maxx=np.max((2,maxx))

    limits=[1e-2,10]
        

    # plot time series
    for idx, ff in enumerate(files):

        if 'job-00' in ff.parent.name:
            jobnum=int(ff.parent.name.split('-')[-1])
        else:
            jobnum=idx


        ax_ts = fig.add_subplot(gs[idx//2 , idx % 2 * (2+1)])

        time=times[idx]
        data=datas[idx]

        ax_ts.plot(time,data,color=color, lw=.1,label=f'potential {jobnum:d}')
        ax_ts.set_xlabel(f'time (ns)')
        ax_ts.set_yscale('log')
        #ax_ts.set_xscale('log')
        ax_ts.grid(which='both',lw=.2, alpha=0.5)
        ax_ts.legend()
        ax_ts.set_ylim(limits)

        ax_hist = fig.add_subplot(gs[idx//2, idx % 2 * (2+1) + 1],sharey=ax_ts)
        ax_hist.hist( data,bins=47,range=limits, orientation='horizontal', color=color, edgecolor='black')
        ax_hist.tick_params(axis='y', labelleft=False)
        ax_hist.set_xticks([])
        ax_hist.set_xlabel(f'count')

        # title
        #ax_ts.set_title(f"{jobnum:003d}")




    

    # adjustments
    fig.suptitle(r'Model deviation')
    plt.tight_layout()

    # write to file
    fname_plot='model_devi.png'
    logger.info(f'saving file {fname_plot}')
    fig.savefig(fname_plot, dpi=300)

#    fname_plot='model_devi_histo.png'
#    logger.info(f'saving file {fname_plot}')
#    fig1.savefig(fname_plot, dpi=300)















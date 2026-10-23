from rich import print
import sys, os
import numpy as np
from logger_setup import logger
from pathlib import Path
from omegaconf import ListConfig
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from ectoolkits_reweight import read_property, read_fes
from utils import find_job_dirs, find_files, join_if_list, extract_field_map,apply_mpl_style, extract_field_map, color_to_color_cmap
import matplotlib
from copy import deepcopy
matplotlib.use('Agg')


def get_cmap(name):
    if '#' in name:
        logger.info(f'generating colormap {name}')
        cmp = color_to_color_cmap("white",name)
    else:
        logger.info(f'getting colormap {name}')
        cmp = plt.cm.get_cmap(name)
    return cmp
        





def run(cfg):
    print(f"[bold green]Running program {cfg._target_.split('.')[1]}:[/bold green] {cfg.param}")


    apply_mpl_style(cfg.mplstyle.dir,cfg.mplstyle.style)
    # get property to plot
    prop=cfg.get('property', None)
    if prop is None:
        print('No poperty to analyze in config file. Exiting.')
        sys.exit()

    # extract relevant cfg
    ##################################################
    logger.info(f'plotting {prop}')
    cfg_fes=cfg.get('surface_data',{}).get('fes',{})
    cfg_prop=cfg.get('surface_data',{}).get(prop,{})
    ppf=cfg_fes.plot
    ppp=cfg_prop.plot
    ##################################################

    # find job dirs and files
    jobdirs=find_job_dirs(cfg.dir_md, cfg.dir_job)

    cvs_selected=cfg.cvs.get('select',None)
    if cvs_selected is None:
        sys.exit(f'you must set cvs.select')
    files_fes=[]
    files_prop=[]
    for jj in jobdirs:
        # find the cvs to look for
        cvs=extract_field_map(jj/'COLVAR').keys()
        cvs=[f for f in cvs if f != 'time' and not f.startswith('opes.')]
        msg=''
        if cvs_selected is not None:
            msg=' from config'
            cvs=cvs_selected
        logger.info(f'selected cvs{msg}: {cvs}')
        ff=find_files(jj, f'pot*-{cvs[0]}-{cvs[1]}-rew-fes.dat')
        fp=find_files(jj, f'pot*-{cvs[0]}-{cvs[1]}-rew-{prop}.dat')
        files_fes+=ff
        files_prop+=fp

    # file average property
    file_ave_fes=find_files(Path(cfg.dir_md), f'{cvs[0]}-{cvs[1]}-rew-fes-AVG.dat')[0]
    file_ave_prop=find_files(Path(cfg.dir_md), f'{cvs[0]}-{cvs[1]}-rew-{prop}-AVG.dat')[0]

    just_average=cfg.get('just_average',False)
    if just_average:
        logger.info('plotting only average FES')
        files_fes=[file_ave_fes]
        files_prop=[file_ave_prop]
    else:
        files_fes.append(file_ave_fes)
        files_prop.append(file_ave_prop)

    # plot settings
    row=ppf.get('row',1)
    col=ppf.get('col',1)
    if len(files_fes)==1 and not 'AVG' in files_fes[0].name:
        row, col = 1,1
    elif just_average:
        row, col = 1,1
    else:
        col=2
        row=len(files_fes)//2 +1


    scale=ppf.get('scale',1)
    size=ppf.get('size',[4,3.5])
    hspace=ppf.get('hspace', 0.3)
    fontsize=ppf.get('fontsize',6)
    plt.rcParams.update({
        "axes.labelsize": fontsize,     # x/y/z axis labels
        "xtick.labelsize": .8*fontsize,    # x tick labels
        "ytick.labelsize": .8*fontsize,    # y tick labels
    })
   
    fig = plt.figure(figsize=(size[0]*scale*col, size[1]*scale*row), dpi=400)
    fig.subplots_adjust(hspace=hspace)
    gs = fig.add_gridspec(row,col) 

    
   # if pp.get('reverse_colormap',False):
   #     my_cmp = my_cmp.reversed()

    # cycle over files

    # plot separate potentials
    for idx, (ff, fp) in enumerate(zip(files_fes, files_prop)):

        if 'job-00' in ff.parent.name:
            jobnum=int(ff.parent.name.split('-')[-1])
        else:
            jobnum=idx

        # data
        logger.info(f'reading file {ff.name}')
        # read fes
        X, Y, Z, Z_extra, _ = read_fes(ff, dim=2) # electronvolts !!!
        if prop != 'fes':
            # read property (grid from fes)
            _ , _, Zp, Zp_extra, _ = read_property(fp, dim=2) 
        else:
            Zp = deepcopy(Z)
            Zp_extra = deepcopy(Z_extra)


        list_of_columns_fes=[Z]
        list_of_columns_prop=[Zp]
        if Z_extra is not None:
            list_of_columns_fes=[Z_extra, Z]
            list_of_columns_prop=[Zp_extra, Zp]


        repeat=0
        while len(list_of_columns_fes)>0:
            if repeat > 0 and just_average:
                break
            # add ax 
            ax  = fig.add_subplot(gs[idx+repeat])
           
            # get arrays
            Z = list_of_columns_fes.pop()
            Zp = list_of_columns_prop.pop()

            ######## 1) contour plot ##########
            # fes clip
            limits_fes=[ppf.get('vmin',None), ppf.get('vmax',None)]
            mm=np.min(Z)
            MM=np.max(Z)
            logger.info(f'property fes is in range [{mm:.3g},{MM:.3g}]')
    
            if limits_fes[0] is not None or limits_fes[1] is not None:
                Z=Z.clip(ppf.vmin, ppf.vmax)
                logger.info(f'clipping property fes to [{ppf.vmin:.3g},{ppf.vmax:.3g}]')
    
            # fes normalization
            if limits_fes[0] is not None:
                mm=limits_fes[0]
            if limits_fes[1] is not None:
                MM=limits_fes[1]
            norm = Normalize(vmin=mm, vmax=MM)
    

            # contour levels
            if ppf.get('contours_fes',True):
                CS = ax.contour(X, Y, Z,levels=ppf.levels, vmin=ppf.vmin, \
                            vmax=ppf.vmax, colors='k', linewidths=0.4,)

            # if not colors add labels to contours
            if not ppp.get('plot_colors',True):
                # colorbar label
                ax.clabel(CS, fontsize=.6*fontsize, inline=True, fmt="%1.2f")

            # set limits
            ax.set_xlim(ppf.limits[0],ppf.limits[1])
            ax.set_ylim(ppf.limits[2],ppf.limits[3])
            
            xlabel=cvs[0] 
            if ppf.xlabel is not None:
                xlabel=join_if_list(ppf.xlabel)
                
            ylabel=cvs[1] 
            if ppf.ylabel is not None:
                ylabel=join_if_list(ppf.ylabel)
            logger.info(f'LABELS are {xlabel} {ylabel}')
    
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            ######## 2) color map ##########
            if ppp.get('plot_colors',True):
                logger.info('plotting color map')

                # prop clip
                mm=np.min(Zp)
                MM=np.max(Zp)
                logger.info(f'property {prop} is in range [{mm:.3g},{MM:.3g}]')
                limits_prop=[ppp.get('vmin',None), ppp.get('vmax',None)]
    
                if limits_prop[0] is not None or limits_prop[1] is not None:
                    Zp=Zp.clip(limits_prop[0], limits_prop[1])
                    logger.info(f'clipping property {prop} to [{ppp.vmin:.3g},{ppp.vmax:.3g}]')
    
                # prop normalization
                if limits_prop[0] is not None:
                    mm=limits_prop[0]
                if limits_prop[1] is not None:
                    MM=limits_prop[1]
                norm = Normalize(vmin=mm, vmax=MM)

                # plot colors
                idx_cmp = 1 if 'AVG' in ff.name else 0
                cmp = get_cmap(ppp.colormap[idx_cmp])
                if 'AVG' in ff.name:
                    if repeat >0:
                        cmp=get_cmap(ppp.colormap[idx_cmp+1])
                        cmp=cmp.reversed()
                if ppp.get('reverse_colormap',False):
                    cmp = cmp.reversed()
                CS2 = ax.contourf(X,Y,Zp,levels=ppp.levels, cmap=cmp, \
                        vmin=mm, vmax=MM, norm=norm,\
                                  )
                if ppp.get('contours',True):
                    color_contour=ppp.get('color_contour','r')
                    CS = ax.contour(X, Y, Zp,levels=ppp.levels, vmin=ppp.vmin, \
                                vmax=ppp.vmax, colors=color_contour, linewidths=0.4,)
        
                # colorbar label
                ax.clabel(CS, fontsize=.6*fontsize, inline=True, fmt="%1.2f")
    
                # colorbar
                cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmp), ax=ax,)
                cbar.set_label(ppp.cbarlabel)
    
            ## mep
            if cfg_prop.plot.get('mep',False):
                if idx < len(jobdirs) and not just_average:
                    fname_mep=f"{cvs[0]}-{cvs[1]}-{prop}-MEP.dat"
                    fmep=find_files(jobdirs[idx], fname_mep, allow_failure=True)
                else:
                    fmep=find_files(cfg.dir_md,f'{cvs[0]}-{cvs[1]}-{prop}-AVG-MEP.dat',allow_failure=True)
                
                if len(fmep)==1:
                    fmep=fmep[0]
                    mep_fes = np.loadtxt(fmep)
                    mep_fes_x = mep_fes[:,0]
                    mep_fes_y = mep_fes[:,1]
                    size_mep_marker=4.0
                    zorder_mep=1000
                    ax.plot(mep_fes_x, mep_fes_y, ls='--', marker='o', markeredgecolor='w',\
                                 markersize=size_mep_marker, markeredgewidth=size_mep_marker/6, label='minimum energy path', color='y', markerfacecolor='y', lw=0.5,
                            zorder=zorder_mep)
                    #start
                    ax.plot(mep_fes_x[0], mep_fes_y[0], ls='', marker='o', markeredgecolor='w',\
                                 markersize=size_mep_marker, markeredgewidth=size_mep_marker/4,  color='g', markerfacecolor='g', lw=0.5,
                            zorder=zorder_mep)
                    #end
                    ax.plot(mep_fes_x[-1], mep_fes_y[-1], ls='', marker='o', markeredgecolor='w',\
                                 markersize=size_mep_marker, markeredgewidth=size_mep_marker/4,  color='r', markerfacecolor='r', lw=0.5,
                            zorder=zorder_mep)
                    # barrier
                    imax=np.argmax(mep_fes[:,2])
                    ax.plot(mep_fes_x[imax], mep_fes_y[imax], ls='', marker='o', markeredgecolor='k',\
                                 markersize=size_mep_marker, markeredgewidth=size_mep_marker/4,  color='y', markerfacecolor='y', lw=0.5,
                            zorder=zorder_mep)
                    #ax.legend()

                    #if 'ave' in ff.name and repeat < 1:
                    #    # reaction energy
                    #    ax.annotate(f'{mep_fes[-1,2]:.2f} eV',
                    #                    xy=(mep_fes_x[-1], mep_fes_y[-1]),
                    #                    xytext=(5, 0),  # 5 points to the right
                    #                    textcoords='offset points',
                    #                    fontsize=6,
                    #                    color='yellow',
                    #                    ha='left',  # horizontal alignment
                    #                    va='center')  # vertical alignment
                    #    # barrier
                    #    ax.annotate(f'{mep_fes[imax,2]:.2f} eV',
                    #                    xy=(mep_fes_x[imax], mep_fes_y[imax]),
                    #                    xytext=(5, 0),  # 5 points to the right
                    #                    textcoords='offset points',
                    #                    fontsize=6,
                    #                    color='yellow',
                    #                    ha='left',  # horizontal alignment
                    #                    va='center')  # vertical alignment

                else:
                    logger.info('skipping mep')


            #if 'AVG' in ff.name and repeat == 0:
            #    cfg_special_pts = cfg.get('surface_data',{}).get(f'{prop}',{}).get('special_points',None)
            #    if cfg_special_pts is not None:
            #        for name, case in cfg_special_pts.get('cases',{}).items():
            #            file_points=case.get('file',None)
            #            assert file_points is not None
            #            fields=extract_field_map(file_points)
            #            data=np.loadtxt(file_points)

            #            xx=data[:,fields[cvs[0]]]
            #            yy=data[:,fields[cvs[1]]]
            #            propdata=data[:,fields[prop]]
            #            propdata=propdata.clip(limits_prop[0], limits_prop[1]) 
            #            sc = ax.scatter(
            #                            xx,
            #                            yy,
            #                            c=propdata,
            #                            cmap=cmp,
            #                            norm=norm,
            #                        )
                        
                    
            mark_points=cfg.get('surface_data',{}).get('points_to_mark',{}).get('draw',False)
            data_points=cfg.get('surface_data',{}).get('points_to_mark',None)

            #print(f'mark_points is {mark_points}')
            #print(f'data_points is {data_points}')
            if mark_points:
                assert data_points is not None
                assert cvs[0] == data_points['units'][0]
                assert cvs[1] == data_points['units'][1]

                for name, series in data_points['data'].items():
                    errors = series.get('errors',None)
                    for idx, pt in enumerate(series['values']):
                        kwargs=series.get('kwargs',{})
                        if errors is not None:
                            err = errors[idx]
                            ax.errorbar(pt[0],pt[1],xerr=err[0], yerr=err[1], zorder=1000,**kwargs)
                        else:
                            ax.scatter(pt[0],pt[1],zorder=1000,**kwargs)

                if data_points.get('legend',False):
                    ax.legend(loc='upper left',fontsize=8,framealpha=0.99,bbox_to_anchor=(0.02, 0.98))
#                    ax.annotate(f'{point[0]:.2f}, {point[1]:.2f}',
#                                        xy=(point[0], point[1]),
#                                        xytext=(5, 5),  # 5 points to the right
#                                        textcoords='offset points',
#                                        fontsize=(int(.8*fontsize)),
#                                        color='yellow',
#                                        ha='left',  # horizontal alignment
#                                        va='center')  # vertical alignment
    
            # title
            tt=ppf.get('title',None)
            if tt is not None:
                ax.set_title(tt)
            elif prop == 'fes' and tt is None:
                tt='FES'
            elif 'AVG' in ff.name and tt is not None:
                ax.set_title(f"{tt}-average")
                if repeat >0:
                    ax.set_title(f"stddev")
        
            repeat+=1
    

    # adjustments
    suptitle=ppf.get('suptitle',None)
    if suptitle is not None:
        fig.suptitle(suptitle)
    adjust=ppf.get('subplots_adjust',None)
    if adjust is not None:
        plt.subplots_adjust(**adjust)
    else:
        fig.tight_layout()

    # write to file
    for ff in list(cfg_prop.save.plot):
        fname_new=f'{cvs[0]}-{cvs[1]}-{prop}{ff}'
        fname_plot=Path(os.getcwd())/fname_new
        logger.info(f'saving file {fname_plot}')
        fig.savefig(fname_plot, dpi=300)















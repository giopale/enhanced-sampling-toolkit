from rich import print
import sys
import numpy as np
from logger_setup import logger
from pathlib import Path
from parser import get_temperature_from_log, parse_state_wfile, parse_bias_factor, parse_colvar_header
from parser import parse_cv_wall
from ectoolkits_reweight import calc_one_FES_point_weights, calc_property_surface, compute_bw_silverman, read_fes, write_fes, read_property
from omegaconf import ListConfig
from utils import extract_field_map, find_job_dirs,find_files, read_colvar_in_time_interval, safe_stack_colvar, apply_mpl_style
import matplotlib.pyplot as plt 




def run(cfg):
    print(f"[bold green]Running program {cfg._target_.split('.')[1]}:[/bold green] {cfg.param}")
    apply_mpl_style(cfg.mplstyle.dir,cfg.mplstyle.style)

    # get relevant config
    #pp=cfg.surface_data.get(prop, {})
    cfg_find=cfg.find_structures

    prop=cfg.get('property', 'fes')
    cvs=cfg.cvs.get('select',None)
    assert cvs is not None

    points_fes=cfg_find.get('points',None)
    assert points_fes is not None
    
    points_fes_names=[f'{cvs[0]}-{pt[0]:.2f}-{cvs[1]}-{pt[1]:.2f}' for pt in points_fes]

    logger.info(f'selecting structures for the following points on the \'{"-".join(cvs)}\' fes: {points_fes}')

    

    # file average property
#     fname_mep=find_files(Path(cfg.dir_md), f'{prop}-{cvs[0]}-{cvs[1]}-rew-ave-mep.dat')[0]
# 
#     #fname_mep='fes-rew-ave-mep.dat'
#     logger.info(f'reading mep from file \'{fname_mep}\'')
#     
#     data_mep=np.loadtxt(fname_mep)
# 
#     mep_start=data_mep[0]
#     mep_ts=data_mep[np.argmax(data_mep[:,-1])]
#     mep_end=data_mep[-1]
#    logger.info(f'mep starting point is [{mep_start[0]:.3f},{mep_start[1]:.3f}]')


#    points_fes=[mep_start[:2], mep_ts[:2],mep_end[:2]]
#    points_fes_names=['START','TS','END']
    

   ###################################### 
    # find directories
    jobdirs=find_job_dirs(Path(cfg.get('dir_md','.')), cfg.dir_job)

    logger.info('#'*100+f' find-structures-bin')



    # loop over directories
    files_written=[]
    for idx, jj in enumerate(jobdirs): 

        logger.info('#'*60+f' {jj.name}')

        # get job index for output
        if 'job-00' in jj.name:
            jj_idx=int(str(jj.name).split('-')[-1])
        else:
            jj_idx=idx


        # get simulation parameters
        temp=get_temperature_from_log(jj/cfg.file_lammps_log)
        state_wfile=parse_state_wfile(jj/cfg.file_plumed_input)
        bias_factor=parse_bias_factor(jj/state_wfile)
        kbT = temp*0.0083144621 # kJ/mol
        nbins=cfg.get('nbins',[101,101])
        
        logger.info(f'kbT is {kbT:.3g}')
        logger.info(f'nbins: {nbins}')
        

        # init data struct
        cvs_dict={}

        # parse cvs from COLVAR 
        colvar_fields=extract_field_map(jj/'COLVAR')

        # make a list of cvs
        cvs=list(cfg.cvs.get('select',None))

#        cvs=[]
#        for field, idx in colvar_fields.items():
#            # filter out timestep
#            if idx == 0:
#                continue
#            # filter out opes 
#            if not field.startswith('opes'):
#                cvs.append(field)
#        msg=f'found {len(cvs):d} cvs: '+ ', '.join(cvs)  
#        logger.info(msg)

        
        # get time interval
        timeint = cfg.get('time_interval',[0.0, np.inf])
        timeint= [1000 * ii if ii is not None else None for ii in timeint] # nanoseconds -> picoseconds 

        # parse COLVAR
        data=read_colvar_in_time_interval(jj/'COLVAR',timeint[0],timeint[1])
    

        # parse CVs not in colvar and add to data
        masks=[]
        msg_config=''
        cv_without_components=''
        for cv_sel in cvs:
            msg_config=' selected from config'
            if not cv_sel in colvar_fields.keys():
                cv_without_components=cv_sel
                if '.' in cv_sel:
                    cv_without_components=cv_sel.split('.')[0]
                logger.info(f'cv {cv_sel} is not in COLVAR. loading file \'CV-{cv_without_components}\'')
                data_new_cv=read_colvar_in_time_interval(jj/f'CV-{cv_without_components}',timeint[0],timeint[1])
                data=safe_stack_colvar(data,data_new_cv)
                colvar_fields.update({cv_sel:np.shape(data)[-1]-1})
                mask=~np.isnan(data_new_cv[:,1])
            else:
                mask=np.array([True]*len(data))
            masks.append(mask)

        # remove elements that have a CV = nan (proton transfer)
        mask_and = masks[0] & masks[1]
        nretain=np.count_nonzero(mask_and)
        if np.count_nonzero(mask_and) != len(data):
            logger.info(f'nan detected. retaining {nretain} structures')
            data=data[mask_and]

        for idx, ii in enumerate(cvs):
            cvs_dict.update({ii:{'data':data[:,colvar_fields[ii]]}})
        # parse opes bias
        bias=data[:, colvar_fields['opes.bias']]



        # parse cvs limits
        for ii in cvs:

            limits=[]
            # get limits from colvar file
            for wall in ['LOWER', 'UPPER']:
                limits.append(parse_cv_wall(file_path=jj/cfg.file_plumed_input,cv=ii,wall=wall))
            limits_cfg=cfg.get('cvs',{}).get(ii,{}).get('limits',[None, None]) # get limits from config or default to the parsed ones

            # override limits with values from config file 
            changed=False
            for kk in [0,1]:
                if limits_cfg[kk] is not None:
                    limits[kk] = limits_cfg[kk]
                    changed=True
            if changed:
                logger.info(f'updating limits: {limits}')
            # store limits into dict 
            cvs_dict.get(ii,{}).update({'limits':limits})
        
        # compute bandwidths Silverman
        sigmas=cfg.get('sigma', [None]*len(cvs))
        weights=np.exp(bias/kbT)
        for idx, ii in enumerate(cvs):
            bw=compute_bw_silverman(cvs_dict[ii]['data'], dim=2, weights=weights, bias_factor=bias_factor)
            bw=bw if sigmas[idx] is None else sigmas[idx]
            cvs_dict[ii]['bandwidth']=bw 
            logger.info(f'bandwidth for \'{ii}\' is {bw:.5f}')
        
        if cfg.get('swap_coordinates',False):
            cvs=cvs.reverse()

        # cycle over selected points in CV space
        for idx, point in enumerate(points_fes):
            logger.info(f'computing weights for point [{point[0]:.3f},{point[1]:.3f}]')
        
            parameters={'cv_x':cvs_dict[cvs[0]]['data'],
                        'cv_y':cvs_dict[cvs[1]]['data'],
                        'bias':bias,
                        'point_x': point[0],
                        'point_y': point[1],
                        'nbins_x':nbins[0],
                        'nbins_y':nbins[1],
                        'sigma_x':cvs_dict[cvs[0]]['bandwidth'],
                        'sigma_y':cvs_dict[cvs[1]]['bandwidth'],
                        'kbT':kbT,
                        'dim':2,
                        }
            
            # get the weights
            # these are the weights relative to the selected point
            weights=calc_one_FES_point_weights(**parameters)
            # normalize
            weights=weights/np.sum(weights)


            # sort and sum
            indices_sorted=np.argsort(weights)[::-1]
            weights_sorted=weights[indices_sorted]
            cum_sum_weights=np.cumsum(weights_sorted)

            # select structures
            logger.info(f'selecting structures constituting {cfg_find.probability_thr*100:.2f}% of the weights, highest weights first')
            prob=cfg_find.probability_thr
            nstruct=np.searchsorted(cum_sum_weights,cfg.find_structures.probability_thr) 
            logger.info(f'{nstruct} structures found, {nstruct/len(weights):.2g}% of total structures')

            # cut the array if the structures are too many
            max_struct=cfg_find.get('max_structures',None)
            if max_struct is None:
                max_struct=np.inf
            if nstruct > max_struct:
                nstruct=max_struct
                logger.info(f'retaining the first {max_struct} structures ({nstruct/len(weights):.2g}% of total),summing up to {cum_sum_weights[nstruct-1]*100:.2f}% weight')
                prob=cum_sum_weights[nstruct-1]
            
    
            indices_struct=indices_sorted[:nstruct]
            weights_struct=weights_sorted[:nstruct]
            cutoff=0.003
            
            if '000' in jj.name:
                #limit=30
                limit=np.where(weights_sorted > cutoff)[-1][-1]

                end=np.max((limit,nstruct))

                fig, ax=plt.subplots()
                yy=weights_sorted[:end+100]
                xx=np.arange(1,len(yy)+1,1)
                ax.plot(xx,yy,ls='-',label='weights', color='tab:blue')

                ax.fill_between(xx[:limit+1],yy[:limit+1],color='tab:blue',alpha=0.3)
                ax.plot([xx[limit+1],ax.get_xlim()[-1]+np.diff(ax.get_ylim())[0]*0.1], [cutoff,cutoff],color='r',lw=.5,label=f'cutoff @ {cutoff:.4f}')

                ax.fill_between(xx[:nstruct-1],yy[:nstruct-1],color='tab:green',alpha=0.1)
                ax.plot([xx[nstruct-1],ax.get_xlim()[-1]+np.diff(ax.get_ylim())[0]*0.1], [yy[nstruct-1],yy[nstruct-1]],color='g',lw=.5,label=f'p={prob*100:.0f}% @ {yy[nstruct-1]:.4f}')

                ax.text(0.1, 0.9, f"nstruct: cutoff/prob {limit+1:d} / {nstruct+1:d}", transform=ax.transAxes, va='bottom',ha='left')
                ax.text(0.1, 0.9, f"$\sum$weights<cutoff: {np.cumsum(weights_sorted)[limit]:.2f}", transform=ax.transAxes, va='top',ha='left')

                ax.set_xlabel('number of structures')
                ax.set_ylabel('weight')
                ax.set_title(f'weights-{points_fes_names[idx]}')
                ax.set_xlim([None,xx[-1]])
                plt.legend()
                fname_plot=f'weights-{points_fes_names[idx]}.png' 
                plt.tight_layout()
            
                plt.savefig(fname_plot)
                logger.info(f'file saved: {fname_plot}')
            
            logger.info(f'first 10 indices: {indices_struct[:10]}')
            sample_freq=100
            timesteps_struct=indices_struct*sample_freq
            logger.info(f'---------------> assuming sample_freq = {sample_freq:d}! <---------------')
            logger.info(f'first 10 TIMESTEPS: {timesteps_struct[:10]}')
            logger.info(f'first 10 WEIGHTS: {weights_struct[:10]}')


            fname_out=f'struct-indices-{points_fes_names[idx]}.dat'
            header=f"# TIMESTEPS of {nstruct} structures, summing up weight to {prob:.4g}, highest weight first"
            line_timesteps=",".join([str(ii) for ii in timesteps_struct])
            line_weights=",".join([f"{ii:.5g}" for ii in weights_struct])
            with open(jj/fname_out, "w") as f:
                f.write(header + "\n")
                f.write(line_weights + "\n")
                f.write(line_timesteps + "\n")
            logger.info(f'indices saved to {jj/fname_out}')

            

            #print(np.argsort(ww)[-5:][::-1])
            #print(len(np.where(ww > 0.003))/len(ww)*100)


#    - cv_x: float, x-coordinate of the collective variable (CV)
#    - cv_y: float, y-coordinate of the collective variable (CV)
#    - bias: float, bias potential value
#    - point_x: float, x-coordinate of the point
#    - point_y: float, y-coordinate of the point
#    - sigma_x: float, standard deviation of the x-coordinate
#    - sigma_y: float, standard deviation of the y-coordinate
#    - kbT: float, product of Boltzmann constant and temperature







# compute mep
# read first (last, TS) point from MEP
# call calc_FES_one_point and retrieve weights array
# one needs to do all the procedure for finding out sigma
# then find out the structures for each potential

# find indices of structures that constitute 99% of the weight



from rich import print
import sys
import os
import numpy as np
from logger_setup import logger
from pathlib import Path

from parser import (
    get_temperature_from_log,
    parse_state_wfile,
    parse_bias_factor,
    parse_colvar_header,
    parse_cv_wall,
)

from ectoolkits_reweight import (
    calc_FES,
    calc_property_surface,
    compute_bw_silverman,
    read_fes,
    write_fes,
    read_property,
    calc_one_FES_point_weights,
)

from omegaconf import ListConfig

from utils import (
    extract_field_map,
    find_job_dirs,
    find_files,
    read_colvar_in_time_interval,
    safe_stack_colvar,
    apply_mpl_style,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd

from utils_reweighting import compute_frame_weights, point_kernel_weights


def safe_filename(name):
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def apply_common_axis_decorations(ax, cfg, xx_type):
    if "cn" in xx_type:
        ax.invert_xaxis()

    lines = cfg.get("cvs_along_mep", {}).get("axvline", [])
    if lines is not None and len(lines) > 0:
        for ll in lines:
            ax.axvline(ll, color="k", lw=0.3, ls="--")


def plot_cv_along_mep(
    ax,
    cv,
    ii,
    xx,
    mask,
    data,
    data_avg,
    data_min,
    data_max,
    cvs_map,
    ccc,
    colors,
    n_jobs,
):
    ppcv = ccc.get("cvs", {}).get(cv, {}).get("plot", {})

    color = ppcv.get("color", colors[ii % len(colors)])
    print(f'FOR CV {cv} COLOR IS {color}')

    logger.info(f"plotting CV along MEP: {cv}")

    for gg in range(n_jobs):
        ax.plot(
            xx,
            data[gg, mask, cvs_map[cv]],
            ls="--",
            lw=0.2,
            color=color,
        )

    ax.plot(
        xx,
        data_avg[mask, cvs_map[cv]],
        ls="-",
        label=cv + " avg",
        color=color,
    )

    ax.fill_between(
        xx,
        data_min[mask, cvs_map[cv]],
        data_max[mask, cvs_map[cv]],
        alpha=0.1,
        color=color,
    )

    pp_axhline = ppcv.get("axhline", None)
    if pp_axhline is not None:
        val = pp_axhline["value"]
        ax.axhline(val, **pp_axhline["kwargs"])

    prop_ax = ppcv.get("prop_ax", {})
    for key, value in prop_ax.items():
        logger.info(f"for CV {cv} setting {key} to {value}")
        setter = getattr(ax, f"set_{key}")
        if isinstance(value, (list, tuple)):
            setter(*value)
        else:
            setter(value)

    ylabel = ppcv.get("ylabel", cv)
    #ax.set_ylabel(ylabel)

    return ax

def plot_mep_along_mep(
    ax,
    xx,
    mask,
    mep_fes,
    meps_dict,
):
    logger.info("plotting MEP")

    ax.set_ylabel("energy (eV)")

    ax.plot(
        xx,
        mep_fes[:, 2],
        color="tab:green",
        label="mep",
        ls="-",
    )

    ax.fill_between(
        xx,
        mep_fes[:, 3],
        mep_fes[:, 4],
        color="tab:green",
        alpha=0.1,
    )

    for tt, job in enumerate(meps_dict.keys()):
        if tt == 0:
            ax.plot(
                xx,
                meps_dict[job][mask],
                color="tab:green",
                label=job,
                ls="--",
                lw=0.2,
                marker="",
                markersize=3,
            )
        else:
            ax.plot(
                xx,
                meps_dict[job][mask],
                color="tab:green",
                label=job,
                ls=":",
                lw=0.2,
                marker="",
                markersize=3,
            )

    return ax



def run(cfg):
    print(
        f"[bold green]Running program {cfg._target_.split('.')[1]}:[/bold green] "
        f"{cfg.param}"
    )

    apply_mpl_style(cfg.mplstyle.dir, cfg.mplstyle.style)
    ccc = cfg.get("cvs_along_mep", {})

    dir_md = cfg.get("dir_md", ".")
    jobdirs = find_job_dirs(dir_md, cfg.dir_job)

    cvs_select = list(cfg.cvs.get("select", None))
    cvs = list(cfg.cvs.get("select", None))

    cvs_add = cfg.cvs.get("add", None)
    if cvs_add is not None:
        cvs_add = list(cvs_add)
        cvs = cvs + cvs_add

    file_mep = find_files("./", Path(f"{cvs[0]}-{cvs[1]}-fes-AVG-MEP.dat"))[0]
    mep = np.loadtxt(file_mep, usecols=[0, 1])

    with open(file_mep, "r") as f:
        header = f.readline()

    meps_dict = {}
    if "job-0" in header:
        column_names = header.replace("#", "").split()
        mep_all = np.loadtxt(file_mep)[:, -4:]
        for idx, job in enumerate(column_names[-4:]):
            meps_dict.update({job: mep_all[:, idx]})

    files_written = []
    dict_results = {}

    for idx, jj in enumerate(jobdirs):
        logger.info("#" * 60 + f" {jj.name}")

        if "job-00" in jj.name:
            jj_idx = int(str(jj.name).split("-")[-1])
        else:
            jj_idx = idx

        temp = get_temperature_from_log(jj / cfg.file_lammps_log)
        state_wfile = parse_state_wfile(jj / cfg.file_plumed_input)
        bias_factor = parse_bias_factor(jj / state_wfile)
        kbT = temp * 0.0083144621

        logger.info(f"kbT is {kbT:.3g}")

        cvs_dict = {}

        colvar_fields = extract_field_map(jj / "COLVAR")
        colvars_plumed = [str(ii) for ii in colvar_fields.keys()]

        msg = "collective variables: " + ", ".join(cvs)
        logger.info(msg)

        timeint = cfg.get("time_interval", [0.0, np.inf])
        timeint = [1000 * ii if ii is not None else None for ii in timeint]
        logger.info(f"selected time interval: {timeint}")

        data = read_colvar_in_time_interval(jj / "COLVAR", timeint[0], timeint[1])

        masks = []

        for cv_sel in cvs:
            scale = cfg.cvs.get(cv_sel, {}).get("scale", 1)

            if cv_sel not in colvar_fields.keys():
                cv_without_components = (
                    cv_sel.split(".")[0] if "." in cv_sel else cv_sel
                )

                logger.info(
                    f"cv {cv_sel} is not in COLVAR. "
                    f"loading file CV-{cv_without_components}"
                )

                data_new_cv = read_colvar_in_time_interval(
                    jj / f"CV-{cv_without_components}",
                    timeint[0],
                    timeint[1],
                )

                if scale != 1:
                    logger.info(f"scaling CV {cv_sel} by {scale:.4g}")
                    data_new_cv[:, 1:] *= scale

                data = safe_stack_colvar(data, data_new_cv)
                colvar_fields.update({cv_sel: np.shape(data)[-1] - 1})

                mask = ~np.isnan(data_new_cv[:, 1])
                n_false = np.count_nonzero(~mask)

                if not np.any(mask):
                    logger.warning(
                        f"nan detected in colvar {cv_sel}: {n_false} cases"
                    )
            else:
                mask = np.array([True] * len(data))

            masks.append(mask)

        mask_new = np.array([True] * len(data))
        cvs_to_filter = cfg["cvs"].get("filter", None)

        if cvs_to_filter is not None:
            for name, limits_filter in cvs_to_filter.items():
                mask_new &= (
                    (data[:, colvar_fields[name]] >= limits_filter[0])
                    & (data[:, colvar_fields[name]] <= limits_filter[1])
                )

        mask_and = mask_new & masks[0] & masks[1]

        if np.count_nonzero(mask_and) != len(data):
            logger.info(
                f"filtering nan/limits, retaining "
                f"{np.count_nonzero(mask_and)} structures "
                f"({np.count_nonzero(mask_and) / len(mask_and) * 100:.3f}%)"
            )
            data = data[mask_and]

        logger.debug("DONE data filtering")

        for ii in cvs:
            logger.debug(f"limits: {ii}")
            limits = []

            for wall in ["LOWER", "UPPER"]:
                limits.append(
                    parse_cv_wall(
                        file_path=jj / cfg["file_plumed_input"],
                        cv=ii,
                        wall=wall,
                    )
                )

            logger.debug("limits: walls done")

            limits_cfg = cfg.get("cvs", {}).get(ii, {}).get("limits", [None, None])
            for kk in [0, 1]:
                if limits_cfg[kk] is not None:
                    limits[kk] = limits_cfg[kk]

            cvs_dict[ii] = {"limits": limits}

            if None in limits:
                logger.info(
                    f"you must set limits for cv '{ii}'. "
                    f"Currently {limits}. Exiting"
                )
                sys.exit()

        logger.debug("DONE parsing cvs limits")

        for key, val in cvs_dict.items():
            logger.info(f"limits for CV {key} set to {val['limits']}")

        for idx_cv, ii in enumerate(cvs):
            cvs_dict[ii]["data"] = data[:, colvar_fields[ii]]

        bias = data[:, colvar_fields["opes.bias"]]

        weights = compute_frame_weights(bias, kbT, normalize=True)

        matrix_kernel = np.zeros((2, 2))

        for idx_cv, ii in enumerate(cvs_select):
            sig = cfg.get("cvs", {}).get(ii, {}).get("sigma", None)

            msg = "(specified from config)"

            if sig is None:
                msg = "(silverman)"
                bw = compute_bw_silverman(
                    cvs_dict[ii]["data"],
                    dim=2,
                    weights=weights,
                    bias_factor=bias_factor,
                )
            else:
                bw = sig

            logger.info(f"bandwidth for '{ii}' is {bw:.5f} {msg}")
            matrix_kernel[idx_cv, idx_cv] = bw**2

        options = {
            "cv_x": cvs_dict[cvs[0]]["data"],
            "cv_y": cvs_dict[cvs[1]]["data"],
            "logw": np.log(weights),
            "H": matrix_kernel,
            "xg": None,
            "yg": None,
            "use_H_inv": False,
            "frame_chunk": 200_000,
            "return_log": False,
        }

        dict_values = {}

        for cv in cvs[2:]:
            values = np.zeros(len(mep))

            for idx, point in enumerate(mep):
                options["xg"] = point[0]
                options["yg"] = point[1]

                weights_on_fes, _ = point_kernel_weights(**options)

                values[idx] = np.average(
                    cvs_dict[cv]["data"],
                    weights=weights_on_fes,
                )

            dict_values.update({cv: values})

        dict_results.update({str(jj): dict_values})

    data = np.zeros((len(jobdirs), len(mep), len(cvs[2:])))

    for ll, (key, jobdata) in enumerate(dict_results.items()):
        for kk, val in enumerate(jobdata.values()):
            data[ll, :, kk] = val

    data_avg = np.mean(data, axis=0)
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data_std = np.std(data, axis=0)

    cvs_map = {cc: idx for idx, cc in enumerate(cvs[2:])}

    file_loc = Path("localization.csv")
    add_loc = 1 if file_loc.is_file() else 0

    row = (len(cvs_add) if cvs_add is not None else 0) + 1 + add_loc
    col = 1

    pp = cfg.get("surface_data", {}).get("fes", {}).plot
    fontsize = pp.get("fontsize", 10)
    logger.info(f'font size: {fontsize}')

    plt.rcParams.update(
        {
            "axes.labelsize": fontsize,
            "xtick.labelsize": 0.8 * fontsize,
            "ytick.labelsize": 0.8 * fontsize,
        }
    )

    scale = pp.get("scale", 1)
    size = cfg.get("size", [4, 3.5])
    hspace = pp.get("hspace", 0.3)

    logger.info("starting matplotlib")

    fig = plt.figure(
        figsize=(size[0] * scale * col, size[1] * scale * row),
        dpi=400,
    )
    fig.subplots_adjust(hspace=hspace)
    gs = fig.add_gridspec(row, col)

    mep_fes = np.loadtxt(file_mep)

    print("cv_x", mep_fes[:, 0])
    print("cv_y", mep_fes[:, 1])
    print("fes", mep_fes[:, 2])

    with open(file_mep, "r") as f:
        header = f.readline()

    column_names = [
        str(ii)
        for ii in header.replace("#", "").replace(" ", "").strip().split(",")
    ]
    field_map = {name: index for index, name in enumerate(column_names)}

    selection = ccc.get("select_by_coordinate", None)

    if selection is not None:
        for key, val in selection.items():
            limits_selection = [-np.inf, np.inf]

            if val[0] is not None:
                limits_selection[0] = val[0]

            if val[1] is not None:
                limits_selection[1] = val[1]

        mask = (
            (mep_fes[:, field_map[key]] >= limits_selection[0])
            & (mep_fes[:, field_map[key]] <= limits_selection[1])
        )
    else:
        mask = np.ones(len(mep_fes), dtype=bool)

    xx_type = ccc.get("plot", {}).get("xaxis", "linear")

    if "linear" in xx_type:
        xx0 = np.linspace(0, 1, len(mep_fes))
        xlabel = "reaction coordinate"
    elif "cn" in xx_type:
        xx0 = mep_fes[:, 1]
        xlabel = "coordination number"
    else:
        xx0 = np.linspace(0, 1, len(mep_fes))
        xlabel = "reaction coordinate"

    mep_fes = mep_fes[mask]
    xx = xx0[mask]

    axes = []

    ax = fig.add_subplot(gs[0])
    
    plot_mep_along_mep(
        ax=ax,
        xx=xx,
        mask=mask,
        mep_fes=mep_fes,
        meps_dict=meps_dict,
    )
    
    axes.append(ax)

    colors = [
        "#e91e63",
        "#9c27b0",
        "#3f51b5",
        "#03a9f4",
        "#4caf50",
        "#ff9800",
        "#795548",
    ]

    if file_loc.is_file():
        logger.info("plotting weights")

        data_loc = pd.read_csv(file_loc)

        ax = fig.add_subplot(gs[1], sharex=ax)
        ax.set_ylabel("weights")

        select = np.array(data_loc["indices"], dtype=int)

        data_loc["localized"] = data_loc.filter(like="loc_").sum(axis=1)

        delocalized = data_loc["deloc"].values
        localized = data_loc["localized"].values
        ohliquid = data_loc["loc_ohliq"].values

        xvalues = xx0
        keep = mask[select]

        ax.plot(
            xvalues[select][keep],
            delocalized[keep],
            color=colors[0],
            label="delocalized",
            ls="-",
            marker=".",
        )
        ax.plot(
            xvalues[select][keep],
            localized[keep],
            color=colors[1],
            label="localized",
            ls="-",
            marker=".",
        )
        ax.plot(
            xvalues[select][keep],
            ohliquid[keep],
            color=colors[2],
            label="loc_ohliq",
            ls="-",
            marker=".",
        )

        axes.append(ax)

    if cvs_add is not None:
        for ii, cv in enumerate(cvs_add):
            ax = fig.add_subplot(gs[ii + 1 + add_loc], sharex=ax)

            plot_cv_along_mep(
                ax=ax,
                cv=cv,
                ii=ii,
                xx=xx,
                mask=mask,
                data=data,
                data_avg=data_avg,
                data_min=data_min,
                data_max=data_max,
                cvs_map=cvs_map,
                ccc=ccc,
                colors=colors,
                n_jobs=len(jobdirs),
            )

            axes.append(ax)

    ax.set_xlabel(xlabel)

    for ax in axes:
        apply_common_axis_decorations(ax, cfg, xx_type)

    points = cfg.get("cvs_along_mep", {}).get("points", None)

    if points is not None:
        for point in points:
            axes[0].scatter(
                2 * point / (len(mep_fes) - 1),
                mep_fes[point, 2],
                color="k",
                lw=0.3,
                marker="h",
                zorder=1000,
            )

    plt.tight_layout()

    for kk in [".png", ".pdf"]:
        fname_out = Path(os.getcwd()) / ("cvs-along-mep" + kk)
        fig.savefig(fname_out)
        logger.info(f"saving file {fname_out}")

    plt.close(fig)

    outdir = Path(os.getcwd()) / "cvs-along-mep-separate"
    outdir.mkdir(exist_ok=True)
    
    logger.info("saving separate MEP plot")
    
    fig_mep, ax_mep = plt.subplots(
        figsize=(size[0] * scale, size[1] * scale),
        dpi=400,
    )
    
    plot_mep_along_mep(
        ax=ax_mep,
        xx=xx,
        mask=mask,
        mep_fes=mep_fes,
        meps_dict=meps_dict,
    )
    
    # ax_mep.set_xlabel(xlabel)
    apply_common_axis_decorations(ax_mep, cfg, xx_type)
    
    points = cfg.get("cvs_along_mep", {}).get("points", None)
    
    if points is not None:
        for point in points:
            ax_mep.scatter(
                2 * point / (len(mep_fes) - 1),
                mep_fes[point, 2],
                color="k",
                lw=0.3,
                marker="h",
                zorder=1000,
            )
    
    adjust = ccc.get("plot", {}).get("subplots_adjust", None)
    if adjust is not None:
        fig_mep.subplots_adjust(**adjust)
    else:
        fig_mep.tight_layout()
    
    for kk in [".png", ".pdf"]:
        fname_out = outdir / f"cvs-along-mep-mep{kk}"
        fig_mep.savefig(fname_out)
        logger.info(f"saving file {fname_out}")
    
    plt.close(fig_mep)

    if cvs_add is not None:
        outdir = Path(os.getcwd()) / "cvs-along-mep-separate"
        outdir.mkdir(exist_ok=True)

        for ii, cv in enumerate(cvs_add):
            logger.info(f"saving separate CV plot: {cv}")

            fig_cv, ax_cv = plt.subplots(
                figsize=(size[0] * scale, size[1] * scale),
                dpi=400,
            )

            plot_cv_along_mep(
                ax=ax_cv,
                cv=cv,
                ii=ii,
                xx=xx,
                mask=mask,
                data=data,
                data_avg=data_avg,
                data_min=data_min,
                data_max=data_max,
                cvs_map=cvs_map,
                ccc=ccc,
                colors=colors,
                n_jobs=len(jobdirs),
            )

            #ax_cv.set_xlabel(xlabel)
            apply_common_axis_decorations(ax_cv, cfg, xx_type)

            adjust=ccc.get('plot',{}).get('subplots_adjust',None)
            if adjust is not None:
                plt.subplots_adjust(**adjust)
            else:
                fig_cv.tight_layout()

            safe_cv = safe_filename(cv)

            for kk in [".png", ".pdf"]:
                fname_out = outdir / f"cvs-along-mep-{safe_cv}{kk}"
                fig_cv.savefig(fname_out)
                logger.info(f"saving file {fname_out}")

            plt.close(fig_cv)

    logger.info("done")

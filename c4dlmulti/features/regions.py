from datetime import datetime, timedelta
from itertools import chain

import dask
import netCDF4
import numpy as np
from scipy.signal import convolve


def load_patches(fn, in_memory=True):
    if in_memory:
        with open(fn, 'rb') as f:
            ds_raw = f.read()
        fn = None
    else:
        ds_raw = None

    with netCDF4.Dataset(fn, 'r', memory=ds_raw) as ds:
        patch_data = {       
            "patches": np.array(ds["patches"]),
            "patch_coords": np.array(ds["patch_coords"]),
            "patch_times": np.array(ds["patch_times"]),
            "zero_patch_coords": np.array(ds["zero_patch_coords"]),
            "zero_patch_times": np.array(ds["zero_patch_times"]),
            "zero_value": ds.zero_value
        }
        if "scale" in ds.variables:
            patch_data["scale"] = np.array(ds["scale"])

    return patch_data


def box_locations(patch_coords, patch_times, 
    zero_patch_coords, zero_patch_times,
    interval=timedelta(minutes=5),
    box_size=(24,8,8)
    ):
    t0 = min(patch_times.min(), zero_patch_times.min())
    t1 = max(patch_times.max(), zero_patch_times.max())
    dt = int(interval.total_seconds())
    t_size = (t1-t0) // dt + 1    

    (i1,j1) = patch_coords.max(axis=0)
    (zi1,zj1) = zero_patch_coords.max(axis=0)
    i1 = max(i1,zi1)+1
    j1 = max(j1,zj1)+1

    loc = np.zeros((t_size,i1,j1), dtype=bool)

    times_coords = chain(
        zip(patch_times, patch_coords),
        zip(zero_patch_times, zero_patch_coords)
    )
    for (t,(i,j)) in times_coords:
        tc = (t-t0) // dt
        loc[tc,i,j] = True

    (dtc,di,dj) = box_size
    kernel = np.ones((di,dj), dtype=np.uint32)
    kernel_size = di*dj
    box_locs = {}
    (gi,gj) = np.mgrid[:loc.shape[1]-di+1,:loc.shape[2]-dj+1]
    for (tc0,lt) in enumerate(loc):
        tc1 = tc0 + dtc
        if tc1 >= loc.shape[0]:
            continue

        f = convolve(lt, kernel, mode='valid', method='direct')
        candidates = (f == kernel_size)
        
        i_list = gi[candidates]
        if len(i_list) == 0:
            continue
        j_list = gj[candidates]

        for (i0,j0) in zip(i_list,j_list):
            i1 = i0 + di
            j1 = j0 + dj
            box = loc[tc0:tc1,i0:i1,j0:j1]
            assert(box.shape[1]==di)
            assert(box.shape[2]==dj)
            if box.all():
                if tc0 not in box_locs:
                    box_locs[tc0] = []
                box_locs[tc0].append((i0,j0))

    return (box_locs, t0)

from datetime import datetime, timedelta
from itertools import chain

from numba import njit, prange
import numpy as np
from tensorflow.keras.utils import Sequence


class BatchGenerator:
    def __init__(self, 
        predictors,
        targets,
        raw,
        coords_by_time,
        primary_raw_var,
        box_size=(8,8),
        timesteps=(12,12),
        batch_size=32,
        interval=timedelta(minutes=5),
        random_seed=None,
        valid_frac=None,
        test_frac=None,
        dataset_block_size=12*24
    ):
        self.batch_size = batch_size
        self.interval = interval
        self.timesteps = timesteps
        self.primary_raw_var = primary_raw_var
        self.rng = np.random.RandomState(seed=random_seed)
        
        # mappings from source variables to predictors and targets
        self.pred_sources = {}
        self.pred_transform = {}
        pred_timeframe = {}
        for (pred_name, pred_data) in predictors.items():
            self.pred_sources[pred_name] = pred_data["source_vars"]
            self.pred_transform[pred_name] = pred_data["transform"]
            pred_timeframe[pred_name] = pred_data.get("timeframe", "past")
        self.pred_names_past = [v for v in predictors.keys() if pred_timeframe[v] == "past"]
        self.pred_names_future = [v for v in predictors.keys() if pred_timeframe[v] == "future"]
        self.pred_names_static = [v for v in predictors.keys() if pred_timeframe[v] == "static"]
        self.target_sources = {}
        self.target_transform = {}
        for (target_name, target_data) in targets.items():
            self.target_sources[target_name] = target_data["source_vars"]
            self.target_transform[target_name] = target_data["transform"]
        self.target_names = list(targets.keys())

        # indices for retrieving source data
        self.raw_batch_index = {}
        self.setup_batch_index(primary_raw_var, raw[primary_raw_var], box_size)
        index_limits = self.raw_batch_index[primary_raw_var].index_limits
        for (raw_name, raw_data) in raw.items():
            if raw_name == primary_raw_var:
                continue
            self.setup_batch_index(raw_name, raw_data, box_size,
                index_limits=index_limits)

        # index for coordinates by time
        self.coords_by_time = {
            t: np.array(coords_by_time[t])
            for t in coords_by_time
        }

        self.time_coords = self.train_valid_test_split(
            valid_frac=valid_frac, test_frac=test_frac,
            block_size=dataset_block_size
        )

        self.fixed_coord = {}
        fixed_coord_times = chain(
            self.time_coords["valid"],
            self.time_coords["test"]
        )
        for t in fixed_coord_times:
            coords = self.coords_by_time[t]
            self.fixed_coord[t] = self.rng.randint(len(coords))

    def setup_batch_index(self, raw_name, raw_data, box_size, index_limits=None):
        time_dim = 0

        # source (raw) variables used for past and future timeframes
        sources_past = set(chain(*[self.pred_sources[v] for v in self.pred_names_past]))
        sources_future = set(chain(*[self.pred_sources[v] for v in self.pred_names_future])) | \
            set(chain(*[self.target_sources[v] for v in self.target_names]))
        sources_static = set(chain(*[self.pred_sources[v] for v in self.pred_names_static]))
        # select longest time dimension needed
        if (raw_name in sources_past) or (raw_name == self.primary_raw_var):
            time_dim = self.timesteps[0]
        if raw_name in sources_future:
            time_dim = max(time_dim, self.timesteps[1])
        if raw_name in sources_static:
            time_dim = max(time_dim, 1)
        if time_dim == 0: # this source is not used so we don't need to index it
            return

        interp = raw_data.get("interpolation", None)
        zero_value = raw_data.get("zero_value", 0)
        missing_value = raw_data.get("missing_value", zero_value)
        if interp is None:
            self.raw_batch_index[raw_name] = PatchIndex(
                raw_data["patches"],
                raw_data["patch_coords"],
                raw_data["patch_times"],
                raw_data["zero_patch_coords"],
                raw_data["zero_patch_times"],
                zero_value=zero_value,
                missing_value=missing_value,
                interval=self.interval,
                box_size=(time_dim,)+box_size,
                index_limits=index_limits,
                static=raw_data.get("static", False)
            )
        else:
            self.raw_batch_index[raw_name] = InterpolatingPatchIndex(
                raw_data["patches"],
                raw_data["patch_coords"],
                raw_data["patch_times"],                
                raw_data["zero_patch_coords"],
                raw_data["zero_patch_times"],
                zero_value=zero_value,
                missing_value=missing_value,
                interval=self.interval,
                box_size=(time_dim,)+box_size,
                index_limits=index_limits,
                method=interp,
                stride=raw_data["stride"]
            )

    def train_valid_test_split(self, valid_frac=None,
        test_frac=None, block_size=None):

        times = np.array(sorted(self.coords_by_time))
        n = len(times)
        
        times_valid = []
        if valid_frac is not None:
            n_valid = valid_frac * n
            while len(times_valid) < n_valid:
                t0 = times[self.rng.randint(len(times))]
                t1 = t0 + block_size
                selection = (t0 <= times) & (times < t1)
                times_valid.extend(list(times[selection]))
                times = times[~selection]
        times_valid = np.array(times_valid)
        
        times_test = []
        if test_frac is not None:
            n_test = test_frac * n
            while len(times_test) < n_test:
                t0 = times[self.rng.randint(len(times))]
                t1 = t0 + block_size
                selection = (t0 <= times) & (times < t1)
                times_test.extend(list(times[selection]))
                times = times[~selection]
        times_test = np.array(times_test)

        self.rng.shuffle(times)
        self.rng.shuffle(times_valid)
        self.rng.shuffle(times_test)

        return {
            "train": times, 
            "valid": times_valid, 
            "test": times_test
        }

    def frame_spatial_coordinates(self, t_batch, dataset="train"):
        i_batch = []
        j_batch = []
        for t in t_batch:
            coords = self.coords_by_time[t]
            if dataset == "train":
                coord_ind = self.rng.randint(len(coords))
            else:
                coord_ind = self.fixed_coord[t]
            (i,j) = coords[coord_ind,:]            
            i_batch.append(i)
            j_batch.append(j)

        return (np.array(i_batch), np.array(j_batch))

    def random_augments(self):
        transpose = bool(self.rng.randint(2))
        flipud = bool(self.rng.randint(2))
        fliplr = bool(self.rng.randint(2))

        return (transpose, flipud, fliplr)

    def augment(self, batch, augments):
        (transpose, flipud, fliplr) = augments

        if transpose:
            batch = batch.transpose((0,1,3,2,4))
        if flipud:
            batch = batch[:,:,::-1,:,:]
        if fliplr:
            batch = batch[:,:,:,::-1,:]
        return batch

    def get_batch(self, t, i, j, var_names, var_sources, transform, num_timesteps):
        raw_data = {}
        for var_name in var_names:
            sources = var_sources[var_name]
            for raw_var in sources:
                if raw_var not in raw_data:
                    raw_data[raw_var] = self.raw_batch_index[raw_var](
                        t, i, j, num_timesteps=num_timesteps)

        batch_data = []
        for var_name in var_names:
            raw_vars = (raw_data[raw_var][:,:num_timesteps,...]
                for raw_var in var_sources[var_name])
            transformed_vars = transform[var_name](*raw_vars)
            has_channels = (len(transformed_vars.shape) == 5)
            if not has_channels:
                transformed_vars = transformed_vars.reshape(transformed_vars.shape + (1,))
            batch_data.append(transformed_vars)

        return batch_data

    def batch(self, idx, dataset="train"):
        t_pred = self.time_coords[dataset][
            idx*self.batch_size:(idx+1)*self.batch_size
        ]
        t_target = t_pred + self.timesteps[0]
        (i,j) = self.frame_spatial_coordinates(t_pred, dataset=dataset)

        pred_batch_past = self.get_batch(t_pred, i, j,
            self.pred_names_past, self.pred_sources, self.pred_transform,
            self.timesteps[0]
        )
        pred_batch_future = self.get_batch(t_target, i, j,
            self.pred_names_future, self.pred_sources, self.pred_transform,
            self.timesteps[1]
        )
        pred_batch_static = self.get_batch(t_pred, i, j,
            self.pred_names_static, self.pred_sources, self.pred_transform, 1
        )
        pred_batch = pred_batch_past + pred_batch_future + pred_batch_static
        target_batch = self.get_batch(t_target, i, j,
            self.target_names, self.target_sources, self.target_transform,
            self.timesteps[1]
        )    

        if dataset == "train":
            augments = self.random_augments()
            pred_batch = [self.augment(b, augments) for b in pred_batch]
            target_batch = [self.augment(b, augments) for b in target_batch]

        return (pred_batch, target_batch)


class BatchSequence(Sequence):
    def __init__(self, batch_gen, dataset="train"):
        super().__init__()
        self.batch_gen = batch_gen
        self.dataset = dataset

    def __len__(self):
        return len(self.batch_gen.time_coords[self.dataset]) // \
            self.batch_gen.batch_size

    def __getitem__(self, idx):
        return self.batch_gen.batch(idx, dataset=self.dataset)

    def on_epoch_end(self):
        self.batch_gen.rng.shuffle(self.batch_gen.time_coords["train"])


class PatchIndex:
    IDX_ZERO = -1
    IDX_MISSING = -2

    def __init__(
        self, patch_data, patch_coords, patch_times,
        zero_patch_coords, zero_patch_times,
        interval=timedelta(minutes=5),
        box_size=(12,8,8), zero_value=0,
        missing_value=0,
        index_limits=None, static=False
    ):
        if (index_limits is None) or static:
            t0 = patch_times.min()
            t1 = patch_times.max()        
            (i1,j1) = patch_coords.max(axis=0)
        else:
            (t0, t1, i1, j1) = index_limits
        self.index_limits = (t0, t1, i1, j1)
        (self.t0, self.t1, self.i1, self.j1) = self.index_limits
        
        self.dt = int(round(interval.total_seconds()))
        self.box_size = box_size
        self.zero_value = zero_value
        self.missing_value = missing_value
        self.patch_data = patch_data
        self.sample_shape = (
            box_size[0],
            self.patch_data.shape[1]*box_size[1],
            self.patch_data.shape[2]*box_size[2]
        )
        self.static = static

        self.patch_index = np.full(
            ((t1-t0)//self.dt + 1, i1+1, j1+1),
            PatchIndex.IDX_MISSING,
            dtype=np.int32
        )
        init_patch_index(self.patch_index, patch_coords,
            patch_times, self.t0, self.dt)
        init_patch_index_zero(self.patch_index, zero_patch_coords,
            zero_patch_times, self.t0, self.dt, PatchIndex.IDX_ZERO)

        self._batch = None

    def _alloc_batch(self, n):
        if (self._batch is None) or (self._batch.shape[0] < n):
            del self._batch
            self._batch = np.zeros((n,)+self.sample_shape, self.patch_data.dtype)
        return self._batch

    def __call__(self, t0_all, i0_all, j0_all, num_timesteps=None):
        n = len(t0_all)
        batch = self._alloc_batch(n)
        if num_timesteps is None:
            num_timesteps = self.box_size[0]
        
        if self.static: # override time coordinate
            t0_all = np.zeros_like(t0_all)
        t1_all = t0_all + num_timesteps
        i1_all = i0_all + self.box_size[1]
        j1_all = j0_all + self.box_size[2]
        bi_size = self.patch_data.shape[1]
        bj_size = self.patch_data.shape[2]

        build_batch(batch, self.patch_data, self.patch_index, 
            t0_all, t1_all, i0_all, i1_all, j0_all, j1_all,
            bi_size, bj_size, self.zero_value, 
            self.missing_value, static=self.static)

        return batch


@njit(parallel=True)
def init_patch_index(patch_index, patch_coords, patch_times, t0, dt):
    for k in prange(patch_coords.shape[0]):
        t = (patch_times[k]-t0) // dt
        if (t < 0) or (t >= patch_index.shape[0]):
            continue
        i = patch_coords[k,0]
        j = patch_coords[k,1]
        patch_index[t,i,j] = k


@njit(parallel=True)
def init_patch_index_zero(patch_index, zero_patch_coords, 
    zero_patch_times, t0, dt, idx_zero):

    for k in prange(zero_patch_coords.shape[0]):
        t = (zero_patch_times[k]-t0) // dt
        if (t < 0) or (t >= patch_index.shape[0]):
            continue
        i = zero_patch_coords[k,0]
        j = zero_patch_coords[k,1]
        patch_index[t,i,j] = idx_zero


class InterpolatingPatchIndex(PatchIndex):
    def __init__(self, *args, stride=12, method='linear', **kwargs):
        super().__init__(*args, **kwargs)
        self.stride = stride
        self.method = method

        times_with_data = (self.patch_index >= 0).any(axis=(1,2))
        self.first_valid_step = np.nonzero(times_with_data)[0][0]
        self._batches = None

    def _alloc_batches(self, n_batches, n_samples):
        shape = (n_samples, n_batches)
        if (self._batches is None) or (self._batches.shape[:2] != shape):
            del self._batches
            self._batches = np.zeros(shape+self.sample_shape[1:], 
                self.patch_data.dtype)
        return self._batches

    def __call__(self, t0_all, i0_all, j0_all, num_timesteps=None):
        n_samples = len(t0_all)
        if num_timesteps is None:
            num_timesteps = self.box_size[0]
        
        # find where the last valid time step is for each batch member
        dt0 = (t0_all - self.first_valid_step) % self.stride
        t0_mod = t0_all - dt0
        t0_mod.clip(0, self.patch_index.shape[0]-1, out=t0_mod)

        # retrieve valid time steps overlapping the search period
        t_steps = range(0, num_timesteps+self.stride+1, self.stride)
        num_steps = len(t_steps)
        batches = self._alloc_batches(num_steps, n_samples)
        t_step = t0_mod
        for i in range(num_steps):
            b = super().__call__(t_step, i0_all, j0_all, num_timesteps=1)
            batches[:,i,...] = b[:,0,...]            
            valid_ind = (t_step + self.stride) < self.patch_index.shape[0]            
            t_step[valid_ind] += self.stride

        # compute returned batch using interpolation
        batch_ip = self._alloc_batch(n_samples)
        interp_batch(batch_ip, batches, t0_all, dt0, self.stride, num_timesteps,
            ip_linear=(self.method=='linear'))

        return batch_ip


# numba can't find these values from PatchIndex
IDX_ZERO = PatchIndex.IDX_ZERO
IDX_MISSING = PatchIndex.IDX_MISSING
@njit(parallel=True)
def build_batch(
    batch, patch_data, patch_index,
    t0_all, t1_all, i0_all, i1_all, j0_all, j1_all,
    bi_size, bj_size, zero_value, missing_value, static=False
):
    for k in prange(t0_all.shape[0]):
        t0 = t0_all[k]
        t1 = t1_all[k]
        i0 = i0_all[k]
        i1 = i1_all[k]
        j0 = j0_all[k]
        j1 = j1_all[k]

        for t in range(t0, t1):
            bt = t-t0
            tt = t0 if static else t
            for i in range(i0, i1):
                bi0 = (i-i0) * bi_size
                bi1 = bi0 + bi_size                
                for j in range(j0, j1):                    
                    ind = int(patch_index[tt,i,j])
                    bj0 = (j-j0) * bj_size
                    bj1 = bj0 + bj_size
                    if ind >= 0:                        
                        batch[k,bt,bi0:bi1,bj0:bj1] = patch_data[ind]
                    elif ind == IDX_ZERO:                        
                        batch[k,bt,bi0:bi1,bj0:bj1] = zero_value
                    elif ind == IDX_MISSING:
                        batch[k,bt,bi0:bi1,bj0:bj1] = missing_value


@njit(parallel=True)
def interp_batch(batch_ip, batches, t0_all, dt0, stride, num_timesteps,
    ip_linear=True):

    n = len(t0_all)
    for k in prange(n):
        t0 = t0_all[k]
        t1 = t0 + num_timesteps
        dt = dt0[k]
        prev_batch_index = 0
        for t in range(t0,t1):                
            bt = t-t0
            prev_batch = batches[:,prev_batch_index,...]
            next_batch = batches[:,prev_batch_index+1,...]
            
            if ip_linear:
                w_next = dt/stride
                w_prev = 1-w_next
                batch_ip[k,bt,:,:] = w_prev*prev_batch[k,:,:] + \
                    w_next*next_batch[k,:,:]
            else:            
                batch_ip[k,bt,:,:] = prev_batch[k,:,:] if \
                    dt < stride/2 else next_batch[k,:,:]

            dt += 1
            if dt >= stride:
                dt = 0
                prev_batch_index += 1

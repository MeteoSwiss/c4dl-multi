import numpy as np
try:
    from ..features import batch
except (ImportError, ModuleNotFoundError):
    pass


def occurrence(batch_gen, dataset="train"):
    """The occurrence of a binary target variable.
    """
    n_occ = 0
    n_total = 0
    num_batches = len(batch_gen.time_coords[dataset]) // \
        batch_gen.batch_size
    for i in range(num_batches):
        if i % 10 == 0:
            print("{}/{}".format(i,num_batches))
        (X,Y) = batch_gen.batch(i)
        n_occ += np.count_nonzero(Y[0])
        n_total += np.prod(Y[0].shape)

    return (n_occ/n_total)


def autocorr(batch_gen, dataset="valid", postproc=None):
    batch_seq = batch.BatchSequence(batch_gen, dataset=dataset)
    Ei = None
    for k in range(len(batch_seq)):
        print(f"{k}/{len(batch_seq)}")
        (X,Y) = batch_seq[k]
        Y = Y[0]
        if postproc is not None:
            Y = postproc(Y)
        if Ei is None:
            num_timesteps = Y.shape[1]
            Ei = np.zeros(num_timesteps)
            Eisq = np.zeros(num_timesteps)
            Ej = np.zeros(num_timesteps)
            Ejsq = np.zeros(num_timesteps)
            Eij = np.zeros(num_timesteps)
            N = np.zeros(num_timesteps, dtype=np.uint64)
        for i in range(num_timesteps):
            Yi = Y[:,i,...].astype(np.float64)
            Yi_sum = Yi.sum()
            Yisq_sum = (Yi**2).sum()
            for j in range(i, num_timesteps):                
                Yj = Y[:,j,...].astype(np.float64)
                Ei[j-i] += Yi_sum
                Eisq[j-i] += Yisq_sum
                Ej[j-i] += Yj.sum()
                Ejsq[j-i] += (Yj**2).sum()
                Eij[j-i] += (Yi*Yj).sum()
                N[j-i] += np.prod(Yi.shape)

    Ei /= N
    Eisq /= N
    Ej /= N
    Ejsq /= N
    Eij /= N
    covij = Eij - Ei*Ej
    sigi = Eisq - Ei**2
    sigj = Ejsq - Ej**2
    corrij = covij / np.sqrt(sigi*sigj)
    return corrij


def corr_length_scale(corr, timestep_len=5.0):
    log_corr = np.log(corr)
    x = np.arange(len(corr)) * timestep_len
    lam = -(log_corr[1:] / x[1:]).mean()
    return 1.0/lam

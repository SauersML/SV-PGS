#!/usr/bin/env python

"""
Markov Chain Monte Carlo (MCMC) sampler for polygenic prediction with continuous shrinkage (CS) priors.

"""


import os
import time
import numpy as np
from scipy import linalg
from numpy import random
import gigrnd

# PRSCS_DEVICE=gpu (baselines-genome addition): the per-block step (the Cholesky factor of D_k + diag(1/psi), the two
# triangular solves and the quadratic form) runs in float64 on the GPU through CuPy. Every random draw is still made on
# the host by numpy's generator, in the same order, and copied to the device, so the chain is the CPU chain up to
# floating-point rounding. Blocks, priors, iterations, burn-in and thinning are unchanged.
GPU = os.environ.get('PRSCS_DEVICE') == 'gpu'
# PRSCS_GIG=vector (baselines-genome addition): psi's GIG draws for all variants at once by gigrnd.gigrnd_vec, the same
# sampler over arrays (same distribution, a different random stream from the per-variant loop)
VECTOR_GIG = os.environ.get('PRSCS_GIG') == 'vector'
if GPU:
    import cupy as cp
    import cupyx.scipy.linalg as cplinalg


def block_step_cpu(ld, psi_blk, beta_mrg_blk, noise, sigma, n):
    """PRS-CS's own block update: returns (beta_blk, beta_blk' dinvt beta_blk)."""
    dinvt = ld+np.diag(1.0/psi_blk.T[0])
    dinvt_chol = linalg.cholesky(dinvt)
    beta_tmp = linalg.solve_triangular(dinvt_chol, beta_mrg_blk, trans='T') + np.sqrt(sigma/n)*noise
    beta_blk = linalg.solve_triangular(dinvt_chol, beta_tmp, trans='N')
    return beta_blk, np.dot(np.dot(beta_blk.T, dinvt), beta_blk)


def block_step_gpu(ld, psi_blk, beta_mrg_blk, noise, sigma, n):
    """block_step_cpu on the device: ld is a CuPy (resident) or numpy (streamed) float64 array; the rest are host arrays."""
    dinvt = cp.asarray(ld)+cp.diag(cp.asarray(1.0/psi_blk.T[0]))
    lower = cp.linalg.cholesky(dinvt)  # dinvt = L L', L = U' of scipy's upper factor U
    beta_tmp = cplinalg.solve_triangular(lower, cp.asarray(beta_mrg_blk), lower=True) + float(np.squeeze(np.sqrt(sigma/n)))*cp.asarray(noise)
    beta_blk = cplinalg.solve_triangular(lower, beta_tmp, lower=True, trans='T')
    quad = cp.dot(cp.dot(beta_blk.T, dinvt), beta_blk)
    return cp.asnumpy(beta_blk), cp.asnumpy(quad)


def mcmc(a, b, phi, sst_dict, n, ld_blk, blk_size, n_iter, n_burnin, thin, chrom, out_dir, beta_std, write_psi, write_pst, seed):
    print('... MCMC ...')
    devices = 1
    if GPU:
        devices = cp.cuda.runtime.getDeviceCount()
        if devices > 1:
            # several GPUs: given psi and sigma the blocks' updates are independent, so each device holds a share of
            # the blocks (greedy by Cholesky cost, size^3) and updates them while the others update theirs; every
            # draw is still made on the host in PRS-CS's order and the quadratic forms are summed in block order
            load = [0.0]*devices; device_of = [0]*len(ld_blk)
            for kk in sorted(range(len(ld_blk)), key=lambda k: -ld_blk[k].shape[0] if ld_blk[k].size else 0):
                if ld_blk[kk].size:
                    device_of[kk] = int(np.argmin(load)); load[device_of[kk]] += float(ld_blk[kk].shape[0])**3
            placed = []
            for kk, blk in enumerate(ld_blk):
                if blk.size:
                    host = blk.get() if isinstance(blk, cp.ndarray) else blk
                    with cp.cuda.Device(device_of[kk]):
                        placed.append(cp.asarray(host))
                else:
                    placed.append(blk)
                ld_blk[kk] = None
            ld_blk = placed
            for d in range(devices):
                with cp.cuda.Device(d):
                    cp.get_default_memory_pool().free_all_blocks()
            import concurrent.futures
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=devices)
            where = 'resident on %d devices (Cholesky cost per device %s)' % (devices, ', '.join('%.2e' % l for l in load))
        else:
            # the blocks stay resident on the device when they fit beside one block's working set (dinvt, its factor
            # and the product); otherwise each is copied to the device when it is updated
            cp.get_default_memory_pool().free_all_blocks()
            free = cp.cuda.Device().mem_info[0]
            largest = max(blk.nbytes for blk in ld_blk)
            if all(isinstance(blk, cp.ndarray) for blk in ld_blk if blk.size):
                where = 'resident (prepared on the device)'
            elif sum(blk.nbytes for blk in ld_blk) + 4*largest < free:
                ld_blk = [cp.asarray(blk) if blk.size else blk for blk in ld_blk]
                where = 'resident'
            else:
                where = 'streamed per update'
        block_step = block_step_gpu
        print('... block updates on the GPU (%s), LD %s ...' % (cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), where))
    else:
        block_step = block_step_cpu
    started = time.time(); block_seconds = 0.0

    # seed
    if seed != None:
        random.seed(seed)

    # derived stats
    beta_mrg = np.array(sst_dict['BETA'], ndmin=2).T
    maf = np.array(sst_dict['MAF'], ndmin=2).T
    n_pst = int((n_iter-n_burnin)/thin)
    p = len(sst_dict['SNP'])
    n_blk = len(ld_blk)

    # initialization
    beta = np.zeros((p,1))
    psi = np.ones((p,1))
    sigma = 1.0
    
    if phi == None:
        phi = 1.0; phi_updt = True
    else:
        phi_updt = False

    if write_pst == 'TRUE':
        beta_pst = np.zeros((p,n_pst))

    beta_est = np.zeros((p,1))
    psi_est = np.zeros((p,1))
    sigma_est = 0.0
    phi_est = 0.0

    # MCMC
    pp = 0
    for itr in range(1,n_iter+1):
        if itr % 100 == 0:
            print('--- iter-' + str(itr) + ' ---')
        if itr in (1, 2, 5, 10) or itr % 100 == 0:
            print('... %d iterations done after %.1f s, %.1f s of them in the block updates ...' % (itr - 1, time.time() - started, block_seconds), flush=True)

        block_started = time.time()
        mm = 0; quad = 0.0
        if devices > 1:
            # the same draws in the same order as the sequential loop, then every device's blocks at once
            tasks = []
            for kk in range(n_blk):
                if blk_size[kk] == 0:
                    continue
                idx_blk = range(mm,mm+blk_size[kk])
                tasks.append((kk, idx_blk, random.randn(len(idx_blk),1)))
                mm += blk_size[kk]

            def run_device(d):
                out = {}
                with cp.cuda.Device(d):
                    for kk, idx_blk, noise in tasks:
                        if device_of[kk] == d:
                            out[kk] = block_step(ld_blk[kk], psi[idx_blk], beta_mrg[idx_blk], noise, sigma, n)
                return out

            results = {}
            for out in pool.map(run_device, range(devices)):
                results.update(out)
            for kk, idx_blk, noise in tasks:
                beta[idx_blk], quad_blk = results[kk]
                quad += quad_blk
        else:
            for kk in range(n_blk):
                if blk_size[kk] == 0:
                    continue
                else:
                    idx_blk = range(mm,mm+blk_size[kk])
                    noise = random.randn(len(idx_blk),1)
                    beta[idx_blk], quad_blk = block_step(ld_blk[kk], psi[idx_blk], beta_mrg[idx_blk], noise, sigma, n)
                    quad += quad_blk
                    mm += blk_size[kk]

        block_seconds += time.time() - block_started
        if GPU:
            # numpy's sums over the variants in place of Python's builtin sum over the rows (a Python loop over all
            # variants); the same sums to rounding
            err = max(n/2.0*(1.0-2.0*np.sum(beta*beta_mrg, axis=0)+quad), n/2.0*np.sum(beta**2/psi, axis=0))
        else:
            err = max(n/2.0*(1.0-2.0*sum(beta*beta_mrg)+quad), n/2.0*sum(beta**2/psi))
        sigma = 1.0/random.gamma((n+p)/2.0, 1.0/err)

        delta = random.gamma(a+b, 1.0/(psi+phi))

        if VECTOR_GIG:
            psi[:, 0] = gigrnd.gigrnd_vec(a-0.5, 2.0*delta[:, 0], n*beta[:, 0]**2/float(np.squeeze(sigma)))
        else:
            for jj in range(p):
                psi[jj] = gigrnd.gigrnd(a-0.5, 2.0*delta[jj], n*beta[jj]**2/sigma)
        psi[psi>1] = 1.0

        if phi_updt == True:
            w = random.gamma(1.0, 1.0/(phi+1.0))
            phi = random.gamma(p*b+0.5, 1.0/((np.sum(delta, axis=0) if GPU else sum(delta))+w))

        # posterior
        if (itr>n_burnin) and (itr % thin == 0):
            beta_est = beta_est + beta/n_pst
            psi_est = psi_est + psi/n_pst
            sigma_est = sigma_est + sigma/n_pst
            phi_est = phi_est + phi/n_pst

            if write_pst == 'TRUE':
                beta_pst[:,[pp]] = beta
                pp += 1

    # convert standardized beta to per-allele beta
    if beta_std == 'FALSE':
        beta_est /= np.sqrt(2.0*maf*(1.0-maf))

        if write_pst == 'TRUE':
            beta_pst /= np.sqrt(2.0*maf*(1.0-maf))


    # write posterior effect sizes
    if phi_updt == True:
        eff_file = out_dir + '_pst_eff_a%d_b%.1f_phiauto_chr%d.txt' % (a, b, chrom)
    else:
        eff_file = out_dir + '_pst_eff_a%d_b%.1f_phi%1.0e_chr%d.txt' % (a, b, phi, chrom)

    with open(eff_file, 'w') as ff:
        if write_pst == 'TRUE':
            for snp, bp, a1, a2, beta in zip(sst_dict['SNP'], sst_dict['BP'], sst_dict['A1'], sst_dict['A2'], beta_pst):
                ff.write(('%d\t%s\t%d\t%s\t%s' + '\t%.6e'*n_pst + '\n') % (chrom, snp, bp, a1, a2, *beta))
        else:
            for snp, bp, a1, a2, beta in zip(sst_dict['SNP'], sst_dict['BP'], sst_dict['A1'], sst_dict['A2'], beta_est):
                ff.write('%d\t%s\t%d\t%s\t%s\t%.6e\n' % (chrom, snp, bp, a1, a2, beta))

    # write posterior estimates of psi
    if write_psi == 'TRUE':
        if phi_updt == True:
            psi_file = out_dir + '_pst_psi_a%d_b%.1f_phiauto_chr%d.txt' % (a, b, chrom)
        else:
            psi_file = out_dir + '_pst_psi_a%d_b%.1f_phi%1.0e_chr%d.txt' % (a, b, phi, chrom)

        with open(psi_file, 'w') as ff:
            for snp, psi in zip(sst_dict['SNP'], psi_est):
                ff.write('%s\t%.6e\n' % (snp, psi))

    # print estimated phi
    if phi_updt == True:
        print('... Estimated global shrinkage parameter: %1.2e ...' % phi_est )

    print('... Done ...')
    # (baselines-genome addition) the posterior means, for validation; PRS-CS's own caller ignores the return value
    return beta_est, psi_est, sigma_est, phi_est



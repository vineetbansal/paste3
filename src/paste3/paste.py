from typing import List, Tuple, Optional
import numpy as np
from anndata import AnnData
import ot
from sklearn.decomposition import NMF
from paste3.helper import (
    intersect,
    kl_divergence_backend,
    to_dense_array,
    extract_data_matrix,
)


def pairwise_align(
    sliceA: AnnData,
    sliceB: AnnData,
    alpha: float = 0.1,
    dissimilarity: str = "kl",
    use_rep: Optional[str] = None,
    G_init=None,
    a_distribution=None,
    b_distribution=None,
    norm: bool = False,
    numItermax: int = 200,
    backend=ot.backend.NumpyBackend(),
    use_gpu: bool = False,
    return_obj: bool = False,
    verbose: bool = False,
    gpu_verbose: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, Optional[int]]:
    """
    Calculates and returns optimal alignment of two slices.

    Args:
        sliceA: Slice A to align.
        sliceB: Slice B to align.
        alpha:  Alignment tuning parameter. Note: 0 <= alpha <= 1.
        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        use_rep: If ``None``, uses ``slice.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``slice.obsm[use_rep]``.
        G_init (array-like, optional): Initial mapping to be used in FGW-OT, otherwise default is uniform mapping.
        a_distribution (array-like, optional): Distribution of sliceA spots, otherwise default is uniform.
        b_distribution (array-like, optional): Distribution of sliceB spots, otherwise default is uniform.
        numItermax: Max number of iterations during FGW-OT.
        norm: If ``True``, scales spatial distances such that neighboring spots are at distance 1. Otherwise, spatial distances remain unchanged.
        backend: Type of backend to run calculations. For list of backends available on system: ``ot.backend.get_backend_list()``.
        use_gpu: If ``True``, use gpu. Otherwise, use cpu. Currently we only have gpu support for Pytorch.
        return_obj: If ``True``, additionally returns objective function output of FGW-OT.
        verbose: If ``True``, FGW-OT is verbose.
        gpu_verbose: If ``True``, print whether gpu is being used to user.

    Returns:
        - Alignment of spots.

        If ``return_obj = True``, additionally returns:

        - Objective function output of FGW-OT.
    """

    # Determine if gpu or cpu is being used
    if use_gpu:
        try:
            import torch
        except ModuleNotFoundError:
            print(
                "We currently only have gpu support for Pytorch. Please install torch."
            )

        if isinstance(backend, ot.backend.TorchBackend):
            if torch.cuda.is_available():
                if gpu_verbose:
                    print("gpu is available, using gpu.")
            else:
                if gpu_verbose:
                    print("gpu is not available, resorting to torch cpu.")
                use_gpu = False
        else:
            print(
                "We currently only have gpu support for Pytorch, please set backend = ot.backend.TorchBackend(). Reverting to selected backend cpu."
            )
            use_gpu = False
    else:
        if gpu_verbose:
            print(
                "Using selected backend cpu. If you want to use gpu, set use_gpu = True."
            )

    # subset for common genes
    common_genes = intersect(sliceA.var.index, sliceB.var.index)
    sliceA = sliceA[:, common_genes]
    sliceB = sliceB[:, common_genes]

    # check if slices are valid
    for s in [sliceA, sliceB]:
        if not len(s):
            raise ValueError(f"Found empty `AnnData`:\n{sliceA}.")

    # Backend
    nx = backend

    # Calculate spatial distances
    coordinatesA = sliceA.obsm["spatial"].copy()
    coordinatesA = nx.from_numpy(coordinatesA)
    coordinatesB = sliceB.obsm["spatial"].copy()
    coordinatesB = nx.from_numpy(coordinatesB)

    if isinstance(nx, ot.backend.TorchBackend):
        coordinatesA = coordinatesA.float()
        coordinatesB = coordinatesB.float()
    D_A = ot.dist(coordinatesA, coordinatesA, metric="euclidean")
    D_B = ot.dist(coordinatesB, coordinatesB, metric="euclidean")

    if isinstance(nx, ot.backend.TorchBackend) and use_gpu:
        D_A = D_A.cuda()
        D_B = D_B.cuda()

    # Calculate expression dissimilarity
    A_X, B_X = (
        nx.from_numpy(to_dense_array(extract_data_matrix(sliceA, use_rep))),
        nx.from_numpy(to_dense_array(extract_data_matrix(sliceB, use_rep))),
    )

    if isinstance(nx, ot.backend.TorchBackend) and use_gpu:
        A_X = A_X.cuda()
        B_X = B_X.cuda()

    if dissimilarity.lower() == "euclidean" or dissimilarity.lower() == "euc":
        M = ot.dist(A_X, B_X)
    else:
        s_A = A_X + 0.01
        s_B = B_X + 0.01
        M = kl_divergence_backend(s_A, s_B)
        M = nx.from_numpy(M)

    if isinstance(nx, ot.backend.TorchBackend) and use_gpu:
        M = M.cuda()

    # init distributions
    if a_distribution is None:
        a = nx.ones((sliceA.shape[0],)) / sliceA.shape[0]
    else:
        a = nx.from_numpy(a_distribution)

    if b_distribution is None:
        b = nx.ones((sliceB.shape[0],)) / sliceB.shape[0]
    else:
        b = nx.from_numpy(b_distribution)

    if isinstance(nx, ot.backend.TorchBackend) and use_gpu:
        a = a.cuda()
        b = b.cuda()

    if norm:
        D_A /= nx.min(D_A[D_A > 0])
        D_B /= nx.min(D_B[D_B > 0])

    # Run OT
    if G_init is not None:
        G_init = nx.from_numpy(G_init)
        if isinstance(nx, ot.backend.TorchBackend):
            G_init = G_init.float()
            if use_gpu:
                G_init.cuda()
    pi, logw = my_fused_gromov_wasserstein(
        M,
        D_A,
        D_B,
        a,
        b,
        G_init=G_init,
        loss_fun="square_loss",
        alpha=alpha,
        log=True,
        numItermax=numItermax,
        verbose=verbose,
        use_gpu=use_gpu,
    )
    pi = nx.to_numpy(pi)
    obj = nx.to_numpy(logw["fgw_dist"])
    if isinstance(backend, ot.backend.TorchBackend) and use_gpu:
        torch.cuda.empty_cache()

    if return_obj:
        return pi, obj
    return pi


def center_align(
    A: AnnData,
    slices: List[AnnData],
    lmbda=None,
    alpha: float = 0.1,
    n_components: int = 15,
    threshold: float = 0.001,
    max_iter: int = 10,
    dissimilarity: str = "kl",
    norm: bool = False,
    random_seed: Optional[int] = None,
    pis_init: Optional[List[np.ndarray]] = None,
    distributions=None,
    backend=ot.backend.NumpyBackend(),
    use_gpu: bool = False,
    verbose: bool = False,
    gpu_verbose: bool = True,
) -> Tuple[AnnData, List[np.ndarray]]:
    """
    Computes center alignment of slices.

    Args:
        A: Slice to use as the initialization for center alignment; Make sure to include gene expression and spatial information.
        slices: List of slices to use in the center alignment.
        lmbda (array-like, optional): List of probability weights assigned to each slice; If ``None``, use uniform weights.
        alpha:  Alignment tuning parameter. Note: 0 <= alpha <= 1.
        n_components: Number of components in NMF decomposition.
        threshold: Threshold for convergence of W and H during NMF decomposition.
        max_iter: Maximum number of iterations for our center alignment algorithm.
        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        norm:  If ``True``, scales spatial distances such that neighboring spots are at distance 1. Otherwise, spatial distances remain unchanged.
        random_seed: Set random seed for reproducibility.
        pis_init: Initial list of mappings between 'A' and 'slices' to solver. Otherwise, default will automatically calculate mappings.
        distributions (List[array-like], optional): Distributions of spots for each slice. Otherwise, default is uniform.
        backend: Type of backend to run calculations. For list of backends available on system: ``ot.backend.get_backend_list()``.
        use_gpu: If ``True``, use gpu. Otherwise, use cpu. Currently we only have gpu support for Pytorch.
        verbose: If ``True``, FGW-OT is verbose.
        gpu_verbose: If ``True``, print whether gpu is being used to user.

    Returns:
        - Inferred center slice with full and low dimensional representations (W, H) of the gene expression matrix.
        - List of pairwise alignment mappings of the center slice (rows) to each input slice (columns).
    """

    # Determine if gpu or cpu is being used
    if use_gpu:
        try:
            import torch
        except ModuleNotFoundError:
            print(
                "We currently only have gpu support for Pytorch. Please install torch."
            )

        if isinstance(backend, ot.backend.TorchBackend):
            if torch.cuda.is_available():
                if gpu_verbose:
                    print("gpu is available, using gpu.")
            else:
                if gpu_verbose:
                    print("gpu is not available, resorting to torch cpu.")
                use_gpu = False
        else:
            print(
                "We currently only have gpu support for Pytorch, please set backend = ot.backend.TorchBackend(). Reverting to selected backend cpu."
            )
            use_gpu = False
    else:
        if gpu_verbose:
            print(
                "Using selected backend cpu. If you want to use gpu, set use_gpu = True."
            )

    if lmbda is None:
        lmbda = len(slices) * [1 / len(slices)]

    if distributions is None:
        distributions = len(slices) * [None]

    # get common genes
    common_genes = A.var.index
    for s in slices:
        common_genes = intersect(common_genes, s.var.index)

    # subset common genes
    A = A[:, common_genes]
    for i in range(len(slices)):
        slices[i] = slices[i][:, common_genes]
    print(
        "Filtered all slices for common genes. There are "
        + str(len(common_genes))
        + " common genes."
    )

    # Run initial NMF
    if dissimilarity.lower() == "euclidean" or dissimilarity.lower() == "euc":
        model = NMF(
            n_components=n_components,
            init="random",
            random_state=random_seed,
            verbose=verbose,
        )
    else:
        model = NMF(
            n_components=n_components,
            solver="mu",
            beta_loss="kullback-leibler",
            init="random",
            random_state=random_seed,
            verbose=verbose,
        )

    if pis_init is None:
        pis = [None for i in range(len(slices))]
        W = model.fit_transform(A.X)
    else:
        pis = pis_init
        W = model.fit_transform(
            A.shape[0]
            * sum(
                [
                    lmbda[i] * np.dot(pis[i], to_dense_array(slices[i].X))
                    for i in range(len(slices))
                ]
            )
        )
    H = model.components_
    center_coordinates = A.obsm["spatial"]

    if not isinstance(center_coordinates, np.ndarray):
        print("Warning: A.obsm['spatial'] is not of type numpy array.")

    # Initialize center_slice
    center_slice = AnnData(np.dot(W, H))
    center_slice.var.index = common_genes
    center_slice.obs.index = A.obs.index
    center_slice.obsm["spatial"] = center_coordinates

    # Minimize R
    iteration_count = 0
    R = 0
    R_diff = 100
    while R_diff > threshold and iteration_count < max_iter:
        print("Iteration: " + str(iteration_count))
        pis, r = center_ot(
            W,
            H,
            slices,
            center_coordinates,
            common_genes,
            alpha,
            backend,
            use_gpu,
            dissimilarity=dissimilarity,
            norm=norm,
            G_inits=pis,
            distributions=distributions,
            verbose=verbose,
        )
        W, H = center_NMF(
            W,
            H,
            slices,
            pis,
            lmbda,
            n_components,
            random_seed,
            dissimilarity=dissimilarity,
            verbose=verbose,
        )
        R_new = np.dot(r, lmbda)
        iteration_count += 1
        R_diff = abs(R - R_new)
        print("Objective ", R_new)
        print("Difference: " + str(R_diff) + "\n")
        R = R_new
    center_slice = A.copy()
    center_slice.X = np.dot(W, H)
    center_slice.uns["paste_W"] = W
    center_slice.uns["paste_H"] = H
    center_slice.uns["full_rank"] = center_slice.shape[0] * sum(
        [
            lmbda[i] * np.dot(pis[i], to_dense_array(slices[i].X))
            for i in range(len(slices))
        ]
    )
    center_slice.uns["obj"] = R
    return center_slice, pis


# --------------------------- HELPER METHODS -----------------------------------


def center_ot(
    W,
    H,
    slices,
    center_coordinates,
    common_genes,
    alpha,
    backend,
    use_gpu,
    dissimilarity="kl",
    norm=False,
    G_inits=None,
    distributions=None,
    verbose=False,
):
    center_slice = AnnData(np.dot(W, H))
    center_slice.var.index = common_genes
    center_slice.obsm["spatial"] = center_coordinates

    if distributions is None:
        distributions = len(slices) * [None]

    pis = []
    r = []
    print("Solving Pairwise Slice Alignment Problem.")
    for i in range(len(slices)):
        p, r_q = pairwise_align(
            center_slice,
            slices[i],
            alpha=alpha,
            dissimilarity=dissimilarity,
            norm=norm,
            return_obj=True,
            G_init=G_inits[i],
            b_distribution=distributions[i],
            backend=backend,
            use_gpu=use_gpu,
            verbose=verbose,
            gpu_verbose=False,
        )
        pis.append(p)
        r.append(r_q)
    return pis, np.array(r)


def center_NMF(
    W,
    H,
    slices,
    pis,
    lmbda,
    n_components,
    random_seed,
    dissimilarity="kl",
    verbose=False,
):
    print("Solving Center Mapping NMF Problem.")
    n = W.shape[0]
    B = n * sum(
        [
            lmbda[i] * np.dot(pis[i], to_dense_array(slices[i].X))
            for i in range(len(slices))
        ]
    )
    if dissimilarity.lower() == "euclidean" or dissimilarity.lower() == "euc":
        model = NMF(
            n_components=n_components,
            init="random",
            random_state=random_seed,
            verbose=verbose,
        )
    else:
        model = NMF(
            n_components=n_components,
            solver="mu",
            beta_loss="kullback-leibler",
            init="random",
            random_state=random_seed,
            verbose=verbose,
        )
    W_new = model.fit_transform(B)
    H_new = model.components_
    return W_new, H_new


def my_fused_gromov_wasserstein(
    M,
    C1,
    C2,
    p,
    q,
    G_init=None,
    loss_fun="square_loss",
    alpha=0.5,
    armijo=False,
    log=False,
    numItermax=200,
    tol_rel=1e-9,
    tol_abs=1e-9,
    use_gpu=False,
    **kwargs,
):
    """
    Adapted fused_gromov_wasserstein with the added capability of defining a G_init (inital mapping).
    Also added capability of utilizing different POT backends to speed up computation.

    For more info, see: https://pythonot.github.io/gen_modules/ot.gromov.html
    """

    p, q = ot.utils.list_to_array(p, q)

    p0, q0, C10, C20, M0 = p, q, C1, C2, M
    nx = ot.backend.get_backend(p0, q0, C10, C20, M0)

    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun)

    if G_init is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = (1 / nx.sum(G_init)) * G_init
        if use_gpu:
            G0 = G0.cuda()

    def f(G):
        return ot.gromov.gwloss(constC, hC1, hC2, G)

    def df(G):
        return ot.gromov.gwggrad(constC, hC1, hC2, G)

    if loss_fun == "kl_loss":
        armijo = True  # there is no closed form line-search with KL

    if armijo:

        def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
            return ot.optim.line_search_armijo(
                cost, G, deltaG, Mi, cost_G, nx=nx, **kwargs
            )
    else:

        def line_search(cost, G, deltaG, Mi, cost_G, **kwargs):
            return solve_gromov_linesearch(
                G, deltaG, cost_G, C1, C2, M=0.0, reg=1.0, nx=nx, **kwargs
            )

    if log:
        res, log = ot.optim.cg(
            p,
            q,
            (1 - alpha) * M,
            alpha,
            f,
            df,
            G0,
            line_search,
            log=True,
            numItermax=numItermax,
            stopThr=tol_rel,
            stopThr2=tol_abs,
            **kwargs,
        )

        fgw_dist = log["loss"][-1]

        log["fgw_dist"] = fgw_dist
        log["u"] = log["u"]
        log["v"] = log["v"]
        return res, log

    else:
        return ot.optim.cg(
            p,
            q,
            (1 - alpha) * M,
            alpha,
            f,
            df,
            G0,
            line_search,
            numItermax=numItermax,
            stopThr=tol_rel,
            stopThr2=tol_abs,
            **kwargs,
        )


def solve_gromov_linesearch(
    G, deltaG, cost_G, C1, C2, M, reg, alpha_min=None, alpha_max=None, nx=None, **kwargs
):
    """
    Solve the linesearch in the FW iterations

    Parameters
    ----------

    G : array-like, shape(ns,nt)
        The transport map at a given iteration of the FW
    deltaG : array-like (ns,nt)
        Difference between the optimal map found by linearization in the FW algorithm and the value at a given iteration
    cost_G : float
        Value of the cost at `G`
    C1 : array-like (ns,ns), optional
        Structure matrix in the source domain.
    C2 : array-like (nt,nt), optional
        Structure matrix in the target domain.
    M : array-like (ns,nt)
        Cost matrix between the features.
    reg : float
        Regularization parameter.
    alpha_min : float, optional
        Minimum value for alpha
    alpha_max : float, optional
        Maximum value for alpha
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    Returns
    -------
    alpha : float
        The optimal step size of the FW
    fc : int
        nb of function call. Useless here
    cost_G : float
        The value of the cost for the next iteration


    .. _references-solve-linesearch:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    if nx is None:
        G, deltaG, C1, C2, M = ot.utils.list_to_array(G, deltaG, C1, C2, M)

        if isinstance(M, int) or isinstance(M, float):
            nx = ot.backend.get_backend(G, deltaG, C1, C2)
        else:
            nx = ot.backend.get_backend(G, deltaG, C1, C2, M)

    dot = nx.dot(nx.dot(C1, deltaG), C2.T)
    a = -2 * reg * nx.sum(dot * deltaG)
    b = nx.sum(M * deltaG) - 2 * reg * (
        nx.sum(dot * G) + nx.sum(nx.dot(nx.dot(C1, G), C2.T) * deltaG)
    )

    alpha = ot.optim.solve_1d_linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)

    # the new cost is deduced from the line search quadratic function
    cost_G = cost_G + a * (alpha**2) + b * alpha

    return alpha, 1, cost_G

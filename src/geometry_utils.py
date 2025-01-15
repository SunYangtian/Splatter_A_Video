import numpy as np
import torch
from pytorch3d.ops import knn_points


########## find connectivity from gs ##########
def cal_connectivity_from_points(points, radius=0.1, K=10, least_edge_num=3, node_radius=None, mode='nn', GraphK=4, adaptive_weighting=True):
     # input: [Nv,3]
     # output: information of edges
     # ii : [Ne,] the i th vert
     # jj: [Ne,] j th vert is connect to i th vert.
     # nn: ,  [Ne,] the n th neighbour of i th vert is j th vert.
    Nv = points.shape[0]

    knn_res = knn_points(points[None], points[None], None, None, K=K+1)
    # Remove themselves
    nn_dist, nn_idx = knn_res.dists[0, :, 1:], knn_res.idx[0, :, 1:]  # [Nv, K], [Nv, K]

    # Make sure ranges are within the radius
    nn_idx[:, least_edge_num:] = torch.where(nn_dist[:, least_edge_num:] < radius ** 2, nn_idx[:, least_edge_num:], - torch.ones_like(nn_idx[:, least_edge_num:]))
    
    nn_dist[:, least_edge_num:] = torch.where(nn_dist[:, least_edge_num:] < radius ** 2, nn_dist[:, least_edge_num:], torch.ones_like(nn_dist[:, least_edge_num:]) * torch.inf)
    if adaptive_weighting:
        weight = torch.exp(-nn_dist / nn_dist.mean())
    elif node_radius is None:
        weight = torch.exp(-nn_dist)
    else:
        nn_radius = node_radius[nn_idx]
        weight = torch.exp(-nn_dist / (2 * nn_radius ** 2))
    weight = weight / weight.sum(dim=-1, keepdim=True)

    ii = torch.arange(Nv)[:, None].cuda().long().expand(Nv, K).reshape([-1])
    jj = nn_idx.reshape([-1])
    nn = torch.arange(K)[None].cuda().long().expand(Nv, K).reshape([-1])
    mask = jj != -1
    ii, jj, nn = ii[mask], jj[mask], nn[mask]

    return ii, jj, nn, weight


def produce_edge_matrix_nfmt(verts: torch.Tensor, edge_shape, ii, jj, nn, device="cuda") -> torch.Tensor:
	"""Given a tensor of verts postion, p (V x 3), produce a tensor E, where, for neighbour list J,
	E_in = p_i - p_(J[n])"""

	E = torch.zeros(edge_shape).to(device)
	E[ii, nn] = verts[ii] - verts[jj]

	return E


def estimate_rotation(source, target, ii, jj, nn, K=10, weight=None, sample_idx=None):
    # input: source, target: [Nv, 3]; ii, jj, nn: [Ne,], weight: [Nv, K]
    # output: rotation: [Nv, 3, 3]
    Nv = len(source)
    source_edge_mat = produce_edge_matrix_nfmt(source, (Nv, K, 3), ii, jj, nn)  # [Nv, K, 3]
    target_edge_mat = produce_edge_matrix_nfmt(target, (Nv, K, 3), ii, jj, nn)  # [Nv, K, 3]
    if weight is None:
        weight = torch.zeros(Nv, K).cuda()
        weight[ii, nn] = 1
        print("!!! Edge weight is None !!!")
    if sample_idx is not None:
        source_edge_mat = source_edge_mat[sample_idx]
        target_edge_mat = target_edge_mat[sample_idx]
    ### Calculate covariance matrix in bulk
    D = torch.diag_embed(weight, dim1=1, dim2=2)  # [Nv, K, K]
    # S = torch.bmm(source_edge_mat.permute(0, 2, 1), target_edge_mat)  # [Nv, 3, 3]
    S = torch.bmm(source_edge_mat.permute(0, 2, 1), torch.bmm(D, target_edge_mat))  # [Nv, 3, 3]
    ## in the case of no deflection, set S = 0, such that R = I. This is to avoid numerical errors
    unchanged_verts = torch.unique(torch.where((source_edge_mat == target_edge_mat).all(dim=1))[0])  # any verts which are undeformed
    S[unchanged_verts] = 0
    
    # t2 = time.time()
    U, sig, W = torch.svd(S)
    R = torch.bmm(W, U.permute(0, 2, 1))  # compute rotations
    # t3 = time.time()

    # Need to flip the column of U corresponding to smallest singular value
    # for any det(Ri) <= 0
    entries_to_flip = torch.nonzero(torch.det(R) <= 0, as_tuple=False).flatten()  # idxs where det(R) <= 0
    if len(entries_to_flip) > 0:
        Umod = U.clone()
        cols_to_flip = torch.argmin(sig[entries_to_flip], dim=1)  # Get minimum singular value for each entry
        Umod[entries_to_flip, :, cols_to_flip] *= -1  # flip cols
        R[entries_to_flip] = torch.bmm(W[entries_to_flip], Umod[entries_to_flip].permute(0, 2, 1))
    # t4 = time.time()
    # print(f'0-1: {t1-t0}, 1-2: {t2-t1}, 2-3: {t3-t2}, 3-4: {t4-t3}')
    return R


def cal_arap_error(nodes_sequence, ii, jj, nn, K=10, weight=None, sample_num=512):
    # input: nodes_sequence: [Nt, Nv, 3]; ii, jj, nn: [Ne,], weight: [Nv, K]
    # output: arap error: float
    Nt, Nv, _ = nodes_sequence.shape
    # laplacian_mat = cal_laplacian(Nv, ii, jj, nn)  # [Nv, Nv]
    # laplacian_mat_inv = invert_matrix(laplacian_mat)
    arap_error = 0
    if weight is None:
        weight = torch.zeros(Nv, K).cuda()
        weight[ii, nn] = 1
    source_edge_mat = produce_edge_matrix_nfmt(nodes_sequence[0], (Nv, K, 3), ii, jj, nn)  # [Nv, K, 3]
    sample_idx = torch.arange(Nv).cuda()
    if Nv > sample_num:
        sample_idx = torch.from_numpy(np.random.choice(Nv, sample_num)).long().cuda()
    else:
        source_edge_mat = source_edge_mat[sample_idx]
    weight = weight[sample_idx]
    try:
        for idx in range(1, Nt):
            # t1 = time.time()
            with torch.no_grad():
                rotation = estimate_rotation(nodes_sequence[0], nodes_sequence[idx], ii, jj, nn, K=K, weight=weight, sample_idx=sample_idx)  # [Nv, 3, 3]
            # Compute energy
            target_edge_mat = produce_edge_matrix_nfmt(nodes_sequence[idx], (Nv, K, 3), ii, jj, nn)  # [Nv, K, 3]
            target_edge_mat = target_edge_mat[sample_idx]
            rot_rigid = torch.bmm(rotation, source_edge_mat[sample_idx].permute(0, 2, 1)).permute(0, 2, 1)  # [Nv, K, 3]
            stretch_vec = target_edge_mat - rot_rigid  # stretch vector
            stretch_norm = (torch.norm(stretch_vec, dim=2) ** 2)  # norm over (x,y,z) space
            arap_error += (weight * stretch_norm).sum()
        arap_error = arap_error / Nt
    except Exception as e:
        print(f"Error in cal_arap_error: {e}")
        arap_error = 0
    return arap_error


def cal_smooth_error(features, ii, jj, nn, K=10, weight=None, sample_num=512):
    # input: features: [Nv, C]; ii, jj, nn: [Ne,], weight: [Nv, K]
    # output: smooth error: float
    Nv, C = features.shape
    if weight is None:
        weight = torch.zeros(Nv, K).cuda()
        weight[ii, nn] = 1
    source_edge_mat = produce_edge_matrix_nfmt(features, (Nv, K, C), ii, jj, nn)  # [Nv, K, C]
    sample_idx = torch.arange(Nv).cuda()
    if Nv > sample_num:
        sample_idx = torch.from_numpy(np.random.choice(Nv, sample_num)).long().cuda()
    else:
        source_edge_mat = source_edge_mat[sample_idx]
    weight = weight[sample_idx]
    smooth_error = torch.abs(weight[...,None] * source_edge_mat[sample_idx]).sum()
    return smooth_error
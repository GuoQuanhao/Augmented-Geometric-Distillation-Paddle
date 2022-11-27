# -*- coding: utf-8 -*-
# Time    : 2020/7/31 16:54
# Author  : Yichen Lu

import paddle
from scipy.linalg import null_space
from paddle.nn import functional as F
from .utils import euclidean_dist


def cosine_criterion(embeddings_a, embeddings_b, *args):
    return F.cosine_embedding_loss(embeddings_a, embeddings_b,
                                   paddle.ones([embeddings_a.shape[0]]))


class CosineLoss(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, outputs_a, outputs_b, *args):
        return cosine_criterion(outputs_a["embedding"], outputs_b["embedding"])


class L1Loss(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, outputs_a, outputs_b, *args):
        return (outputs_a["embedding"] - outputs_b['embedding']).abs().sum(axis=1).mean()


class L2Loss(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, outputs_a, outputs_b, *args):
        return (outputs_a["embedding"] - outputs_b['embedding']).pow(2).sum(axis=1).mean()


def pairwise_residual(embeddings_a, embeddings_b, targets, return_flags=False):
    B, *_ = embeddings_a.shape

    residual_a = embeddings_a.unsqueeze(axis=0) - embeddings_a.unsqueeze(axis=1)
    residual_b = embeddings_b.unsqueeze(axis=0) - embeddings_b.unsqueeze(axis=1)

    is_pos = targets.expand([B, B]).t().equal(targets.expand([B, B]))
    assert is_pos.sum(axis=1).astype(paddle.int32).unique().shape[0] == 1, "No CK sampler."
    is_pos = paddle.triu(is_pos, diagonal=1).astype(paddle.bool)
    if return_flags:
        return residual_a[is_pos], residual_b[is_pos], is_pos
    return residual_a[is_pos], residual_b[is_pos]


def triangle_criterion(embeddings_a, embeddings_b, targets, *args):
    embedding_distillation_loss = cosine_criterion(embeddings_a, embeddings_b)

    residual_a, residual_b = pairwise_residual(embeddings_a, embeddings_b, targets)
    residual_cos = (F.normalize(residual_a, p=2, axis=1) * F.normalize(residual_b, p=2, axis=1)).sum(axis=1)

    residual_distillation_loss = 1. - residual_cos.mean()

    return embedding_distillation_loss + 0.5 * residual_distillation_loss


class NoDistillationLoss(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, *args, **kwargs):
        return 0


class ClassificationDistillationLoss(object):
    def __init__(self, config):
        self.T = config["T"]

    def __call__(self, outputs, outputs_target, *args):
        logits, logits_target = outputs['preds'], outputs_target['preds']
        B, C = logits_target.shape
        logits = logits[:, -C:]
        logits_hat = logits / self.T
        logits_target_hat = logits_target.detach() / self.T
        p_hat = F.softmax(logits_hat, axis=1)
        p_target_hat = F.softmax(logits_target_hat, axis=1)
        distillation_loss = F.kl_div(p_hat.log(), p_target_hat, reduction="batchmean")
        return distillation_loss


class MetricDistillationLoss(object):
    def __init__(self, config):
        self.T = config["T"]

    def __call__(self, outputs, outdated_outputs, memory_bank=None, outdated_memory_bank=None):
        embeddings, outdated_embeddings = outputs["embedding"], outdated_outputs["embedding"]
        memory_bank = embeddings if memory_bank is None else paddle.cat([embeddings, memory_bank], axis=0)
        outdated_memory_bank = outdated_embeddings if outdated_memory_bank is None else paddle.cat(
            [outdated_embeddings, outdated_memory_bank], axis=0)

        dist = euclidean_dist(embeddings, memory_bank)
        dist_hat = dist / self.T
        dist_hat = dist_hat + paddle.zeros_like(dist_hat).fill_diagonal_(float('inf'))
        outdated_dist = euclidean_dist(outdated_embeddings, outdated_memory_bank)
        outdated_dist_hat = outdated_dist / self.T
        outdated_dist_hat = outdated_dist_hat + paddle.zeros_like(outdated_dist_hat).fill_diagonal_(float('inf'))

        distribution = F.softmin(dist_hat, axis=1).clamp_min(1e-8)
        outdated_distribution = F.softmin(outdated_dist_hat, axis=1).clamp_min(1e-8)
        distillation_loss = (-1. * outdated_distribution * distribution.log()).sum(axis=1).mean()
        return distillation_loss


class PoolingDistillationLoss(object):
    def __init__(self, config):
        collapse, criterion = config["collapse"], config["criterion"]
        self.embedding_factor, self.attention_factor = config["embedding_factor"], config["attention_factor"]

        self.collapse_func = {"spatial": self.spatial_pooling,
                              "channel": self.channel_pooling,
                              "vertical": self.vertical_pooling,
                              "horizon": self.horizon_pooling,
                              "global": self.global_pooling,
                              "none": self.no_pooling}[collapse]

        self.criterion = {"cosine": cosine_criterion,
                          "triangle": TriangleDistillationLoss(),
                          "res-triangle": ResTriangleDistillationLoss(),
                          "sim-res-triangle": SimResTriangleDistillationLoss()}[criterion]

    def __call__(self, outputs_a, outputs_b, targets=None):
        embedding_a, embedding_b = outputs_a['embedding'], outputs_b['embedding'].detach()
        embedding_distillation_loss = F.cosine_embedding_loss(embedding_a, embedding_b,
                                                              paddle.ones([embedding_a.shape[0]]))

        feature_maps_a, feature_maps_b = [attention.pow(2) for attention in outputs_a['maps']], \
                                         [activation_map.pow(2) for activation_map in outputs_b['maps']]
        feature_map_distillation_losses = [self.pooling_loss(feature_map_a, feature_map_b, targets)
                                           for feature_map_a, feature_map_b in zip(feature_maps_a, feature_maps_b)]
        feature_map_distillation_loss = sum(feature_map_distillation_losses) / len(feature_map_distillation_losses)

        return self.embedding_factor * embedding_distillation_loss + \
               self.attention_factor * feature_map_distillation_loss

    def pooling_loss(self, map_a, map_b, targets=None):
        pooling_vector_a = F.normalize(self.collapse_func(map_a), p=2, axis=1)
        pooling_vector_b = F.normalize(self.collapse_func(map_b), p=2, axis=1)
        return self.criterion(pooling_vector_a, pooling_vector_b, targets)

    @staticmethod
    def channel_pooling(feature_map):
        assert len(feature_map.shape) == 4
        bs = feature_map.shape[0]
        return feature_map.sum(axis=1).reshape([bs, -1])

    @staticmethod
    def vertical_pooling(feature_map):
        assert len(feature_map.shape) == 4
        bs = feature_map.shape[0]
        return feature_map.sum(axis=2).reshape([bs, -1])

    @staticmethod
    def horizon_pooling(feature_map):
        assert len(feature_map.shape) == 4
        bs = feature_map.shape[0]
        return feature_map.sum(axis=3).reshape([bs, -1])

    @staticmethod
    def global_pooling(feature_map):
        assert len(feature_map.shape) == 4
        bs = feature_map.shape[0]
        return feature_map.mean(axis=(2, 3)).reshape([bs, -1])

    @staticmethod
    def spatial_pooling(feature_map):
        assert len(feature_map.shape) == 4
        bs = feature_map.shape[0]
        return paddle.cat([feature_map.sum(axis=2).reshape([bs, -1]), feature_map.sum(axis=3).reshape([bs, -1])], axis=1)

    @staticmethod
    def no_pooling(feature_map):
        assert len(feature_map.shape) == 4
        bs = feature_map.shape[0]
        return feature_map.reshape([bs, -1])


class RelativeDistancesLoss(object):
    """Distillation loss between the teacher and the student comparing distances
        instead of embeddings.
        Reference:
            * Lu Yu et al.
              Learning Metrics from Teachers: Compact Networks for Image Embedding.
              CVPR 2019.
        :param embeddings_a: ConvNet embeddings of a model.
        :param embeddings_b: ConvNet embeddings of a model.
        :return: A float scalar loss.
        """

    def __init__(self, config):
        self.normalize = config.get("normalize", False)
        self.p = config.get("p", 2)

    def __call__(self, outputs_a, outputs_b, *args):
        embeddings_a, embeddings_b = outputs_a["embedding"], outputs_b["embedding"]
        if self.normalize:
            embeddings_a = F.normalize(embeddings_a, axis=-1, p=2)
            embeddings_b = F.normalize(embeddings_b, axis=-1, p=2)

        pairwise_distances_a = paddle.pdist(embeddings_a, p=self.p)
        pairwise_distances_b = paddle.pdist(embeddings_b, p=self.p)

        return (pairwise_distances_a - pairwise_distances_b).abs().mean()


class GradCAMLoss(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, activation_s, grads_s, activation_t, grads_t):
        attention_s = F.adaptive_avg_pool2d(grads_s, (1, 1)) * activation_s
        # attention_s = activation_s
        # attention_s = F.relu(attention_s.sum(axis=1))
        attention_s = F.relu(attention_s)
        attention_t = F.adaptive_avg_pool2d(grads_t, (1, 1)) * activation_t
        # attention_t = activation_t
        # attention_t = F.relu(attention_t.sum(axis=1))
        attention_t = F.relu(attention_t)

        normalized_s = F.normalize(attention_s.reshape([attention_s.shape[0], -1]), p=2, axis=1)
        normalized_t = F.normalize(attention_t.reshape([attention_t.shape[0], -1]), p=2, axis=1)

        loss = paddle.abs(normalized_s - normalized_t).sum(axis=1).mean()
        return loss


class TriangleDistillationLoss(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, outputs_a, outputs_b, targets):
        embeddings_a, embeddings_b = outputs_a["embedding"], outputs_b["embedding"]
        return triangle_criterion(embeddings_a, embeddings_b, targets)


class ResTriangleDistillationLoss(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, outputs_a, outputs_b, targets):
        embeddings_a, embeddings_b = outputs_a["embedding"], outputs_b["embedding"]
        residual_a, residual_b, is_pos = pairwise_residual(embeddings_a, embeddings_b, targets, return_flags=True)

        return triangle_criterion(residual_a,
                                  residual_b,
                                  targets.expand([targets.shape[0], targets.shape[0]])[is_pos]
                                  )


class SimResTriangleDistillationLoss(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, outputs_a, outputs_b, targets):
        embeddings_a, embeddings_b = outputs_a["embedding"], outputs_b["embedding"]
        return cosine_criterion(*pairwise_residual(embeddings_a, embeddings_b, targets))


class SimilitudeDistillationLoss(object):
    def __init__(self, *args):
        self.args = args

    def __call__(self, outputs_a, outputs_b, pids):
        embeddings_a, embeddings_b = outputs_a["embedding"], outputs_b["embedding"]
        B, *_ = embeddings_a.shape
        is_pos = pids.expand([B, B]).t().equal(pids.expand([B, B]))
        dist_s = euclidean_dist(embeddings_a, embeddings_a)
        dist_t = euclidean_dist(embeddings_b, embeddings_b)
        scales = (dist_s / dist_t)[paddle.triu(is_pos, diagonal=1).astype(paddle.bool)].reshape([pids.unique().shape[0], -1])

        # scales_prob = scales.softmax(axis=1)
        # loss = (scales_prob * scales_prob.log()).sum(axis=1).mean()

        loss = scales.var(axis=1).mean()
        return loss


def sqrt_newton_schulz_autograd(A, numIters=2):
    # batchSize = A.data.shape[0]
    dim = A.data.shape[0]
    normA = A.mul(A).sum(axis=0).sum(axis=0).sqrt()
    Y = A.div(normA.reshape([1, 1]).expand_as(A))
    I = paddle.eye(dim, dim).astype(paddle.float64)
    Z = paddle.eye(dim, dim).astype(paddle.float64)

    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)
    sA = Y * paddle.sqrt(normA).expand_as(A)
    return sA


class GFK(object):
    def __init__(self, config):
        self.dim = config["dim"]
        self.eps = 1e-2

    def decompose(self, inputs, subspace_dim=2):
        square_mat = inputs.transpose(0, 1).mm(inputs)
        sA_minushalf = self.sqrt_newton_schulz_minus(square_mat, numIters=1)
        ortho_mat = inputs.astype(paddle.float64).mm(sA_minushalf)
        ortho_mat = ortho_mat.astype(paddle.float32)

        return ortho_mat[:, :subspace_dim]

    @staticmethod
    def train_pca_tall(data, subspace_dim):
        """
        Modified PCA function, different from the one in sklearn
        :param data: data matrix
        :param subspace_dim: dim
        :return: a wrapped machine object
        """

        data2 = data - data.mean(0)
        uu, ss, vv = paddle.svd(data2.astype(paddle.float32))
        subspace = uu[:, :subspace_dim]

        return subspace

    @staticmethod
    def sqrt_newton_schulz_minus(A, numIters=1):
        # batchSize = A.data.shape[0]
        A = A.astype(paddle.float64)
        dim = A.data.shape[0]
        normA = A.mul(A).sum(axis=0).sum(axis=0).sqrt()
        Y = A.div(normA.reshape([1, 1]).expand_as(A))
        I = paddle.eye(dim, dim).astype(paddle.float64)
        Z = paddle.eye(dim, dim).astype(paddle.float64)

        # A.register_hook(print)
        for i in range(numIters):
            T = 0.5 * (3.0 * I - Z.mm(Y))
            Y = Y.mm(T)
            Z = T.mm(Z)

        sZ = Z * 1. / paddle.sqrt(normA).expand_as(A)
        return sZ

    def fit(self, input1, input2):
        """
        Obtain the kernel G
        :param input1: ns * n_feature, source feature
        :param input2: nt * n_feature, target feature
        :return: GFK kernel G
        """

        input1 = F.normalize(input1, axis=-1, p=2)
        input2 = F.normalize(input2, axis=-1, p=2)

        source_dim = input1.shape[0] - 2  # min(64, input1.shape[0]-2)#input1.shape[0]//2#-2
        target_dim = input2.shape[0] - 2  # min(64, input2.shape[0]-2)#16#input2.shape[0]//2#-2
        num_nullspacedim = 60

        # PSRS
        # source = supportset.contiguous().reshape([-1, supportset.shape[-1]]))#[:source_dim]
        source = input1
        Ps = self.train_pca_tall(source.t(), subspace_dim=source_dim)  # .detach()
        Rs = paddle.from_numpy(null_space(Ps.t().cpu().detach().numpy())[:, :num_nullspacedim])
        # adding columns
        Ps = paddle.cat([Ps, Rs], axis=1)
        N = Ps.shape[1]  # L = NxK shot - 1

        target = input2
        Pt = self.train_pca_tall(target.t(), subspace_dim=target_dim)

        # Pt.register_hook(print)

        G = self.gfk_G(Ps, Pt, N, source_dim, target_dim).detach().astype(paddle.float32)

        # G.register_hook(print)
        # sqrG = self.sqrt_newton_schulz_autograd(G.astype(paddle.float64), numIters=1).astype(paddle.float32)
        # qq = query #- supportset[ii].mean(0)
        # meann = supportset[ii].mean(0)

        qq1 = input1 - input2
        # qq1_norm = input1/(paddle.norm(input1, axis=-1, keepdim=True) + 1e-18)
        # qq2_norm = input2/(paddle.norm(input2, axis=-1, keepdim=True) + 1e-18)
        projected_qq = G.t().mm(qq1.t()).t()

        projected_qq_norm = projected_qq  # /(paddle.norm(projected_qq, axis=-1, keepdim=True) + 1e-18)
        # loss =  paddle.sum((qq1_norm) * projected_qq_norm, axis=-1)
        loss = paddle.sum((qq1) * projected_qq_norm, axis=-1)  # *1e-1
        # ones = paddle.ones_like(loss)
        loss_kd = loss.mean()  # paddle.mean(ones - loss)#loss.mean()#
        # new_query_dist =paddle.sqrt(paddle.sum(new_query_dist*new_query_dist, axis=-1) + 1e-10)#paddle.sum(new_query_dist*new_query_dist, axis=-1)#*0.2#

        return loss_kd

    def gfk_G(self, Ps, Pt, N, source_dim, target_dim):
        A = Ps[:, :source_dim].t().mm(Pt)  # QPt[:source_dim, :]#.copy()
        B = Ps[:, source_dim:].t().mm(Pt)  # QPt[source_dim:, :]#.copy()

        ######## GPU #############

        UU, SS, VV = self.HOGSVD_fit([A, B])
        # SS.register_hook(print)
        V1, V2, V, Gam, Sig = UU[0], UU[1], VV, SS[0], SS[1]
        V2 = -V2

        Gam = Gam.clamp(min=-1., max=1.)
        theta = paddle.acos(Gam)  # + 1e-5

        B1 = paddle.diag(0.5 * (1 + (paddle.sin(2 * theta) / (2. * theta + 1e-12))))
        B2 = paddle.diag(0.5 * (paddle.cos(2 * theta) - 1) / (2 * theta + 1e-12))
        B3 = B2
        B4 = paddle.diag(0.5 * (1. - (paddle.sin(2. * theta) / (2. * theta + 1e-12))))

        delta1_1 = paddle.cat((V1, paddle.zeros((N - source_dim, target_dim))),
                             axis=0)  # np.hstack((V1, paddle.zeros((dim, N - dim))))
        delta1_2 = paddle.cat((paddle.zeros((source_dim, target_dim)), V2),
                             axis=0)  # np.hstack((np.zeros(shape=(N - dim, dim)), V2))

        delta1 = paddle.cat((delta1_1, delta1_2), axis=1)

        delta2_1 = paddle.cat((B1, B3), axis=0)  # c
        delta2_2 = paddle.cat((B2, B4), axis=0)  #
        delta2 = paddle.cat((delta2_1, delta2_2), axis=1)

        delta3_1 = paddle.cat((V1.t(), paddle.zeros((target_dim, source_dim))),
                             axis=0)  # np.hstack((V1, np.zeros(shape=(dim, N - dim))))
        delta3_2 = paddle.cat((paddle.zeros((target_dim, N - source_dim)), V2.t()),
                             axis=0)  # np.hstack((np.zeros(shape=(N - dim, dim)), V2))
        delta3 = paddle.cat((delta3_1, delta3_2), axis=1)  # .t()  # np.vstack((delta3_1, delta3_2)).T

        mm_delta = paddle.matmul(delta1, delta2)

        delta = paddle.matmul(mm_delta, delta3)
        G = paddle.matmul(paddle.matmul(Ps, delta), Ps.t()).astype(paddle.float32)

        return G

    ############################## HOGSVD #########################
    def inverse(self, X):
        eye = paddle.diag(paddle.randn(X.shape[0])).astype(paddle.float64) * self.eps
        # X = X + eye
        # A = paddle.inverse(X)
        Z = self.sqrt_newton_schulz_minus(X.astype(paddle.float64), numIters=1).astype(paddle.float32)
        A = Z.mm(Z)  ## inverse
        # A[0].register_hook(A)
        return A.astype(paddle.float32)

    def HOGSVD_fit_S(self, X):
        N = len(X)
        data_shape = X[0].shape
        # eye = paddle.diag(paddle.randn(data_shape[1])) * self.eps
        # A = [x.T.dot(x) for x in X]
        A = [paddle.matmul(x.transpose(0, 1), x).astype(paddle.float32) for x in X]
        A_inv = [self.inverse(a.astype(paddle.float64)).astype(paddle.float32) for a in A]
        S = paddle.zeros((data_shape[1], data_shape[1])).astype(paddle.float32)
        for i in range(N):
            for j in range(i + 1, N):
                S = S + (paddle.matmul(A[i], A_inv[j]) + paddle.matmul(A[j], A_inv[i]))
        S = S / (N * (N - 1))
        # S.register_hook(print)
        return S

    def _eigen_decompostion(self, X, subspace_dim):
        V, eigen_values, V_t = paddle.svd(X.astype(paddle.float64))
        # V = self.decompose(X.t(), subspace_dim=subspace_dim)
        return V.astype(paddle.float32)

    def HOGSVD_fit_B(self, X, V):
        X = [x.astype(paddle.float32) for x in X]
        # V.register_hook(print)
        V_inv = V.t()  # V_inv is its transpose #paddle.inverse(V).astype(paddle.float32)#self.inverse(V).astype(paddle.float32)  # paddle.inverse(V)
        # V_inv.register_hook(print)
        B = [paddle.matmul(V_inv, x.transpose(0, 1)).transpose(0, 1) for x in X]
        # B[0].register_hook(print)
        return B

    def HOGSVD_fit_U_Sigma(self, B):
        B = [b for b in B]
        sigmas = paddle.stack([paddle.norm(b, axis=0) for b in B])
        # B[0].register_hook(print)
        U = [b / (sigma) for b, sigma in zip(B, sigmas)]

        return sigmas, U

    def HOGSVD_fit(self, X):
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : array-like, shape (n_samples, (n_rows_i, n_cols)
            List of training input samples. Eah input element has
            the same numbe of columns but can have unequal number of rows.
        Returns
        -------
        self : object
            Returns self.
        """

        X = [x for x in X]

        # Step 1: Calculate normalized S
        S = self.HOGSVD_fit_S(X).astype(paddle.float32)
        # S.register_hook(print)

        V = self._eigen_decompostion(S, S.shape[0])

        B = self.HOGSVD_fit_B(X, V)

        sigmas, U = self.HOGSVD_fit_U_Sigma(B)

        return U, sigmas, V

    def __call__(self, outputs_a, outputs_b, *args):
        embeddings_a, embeddings_b = outputs_a["embedding"], outputs_b["embedding"]
        return self.fit(embeddings_a, embeddings_b)


class GeoDLxLUCIR(object):
    def __init__(self, config):
        self.config = config
        self.geoDL_criterion = GFK(config)

    def __call__(self, outputs_a, outputs_b, *args):
        embeddings_a, embeddings_b = outputs_a["embedding"], outputs_b["embedding"]
        geoDL_loss = self.geoDL_criterion.fit(embeddings_a, embeddings_b)
        cosine_loss = cosine_criterion(embeddings_a, embeddings_b)
        loss = self.config["GeoDL_factor"] * geoDL_loss + self.config["cosine_factor"] * cosine_loss
        return loss


class AlwaysBeDreaming(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, outputs_s, outputs_t, *args):
        outdated_preds, preds = outputs_s["outdated_preds"], outputs_t["preds"]
        loss = F.mse_loss(outdated_preds, preds)
        return loss


factory = {
    "cosine": CosineLoss,
    "triangle": TriangleDistillationLoss,
    "res-triangle": ResTriangleDistillationLoss,
    "sim-res": SimResTriangleDistillationLoss,
    "no": NoDistillationLoss,
    "hinton": ClassificationDistillationLoss,
    "metric": MetricDistillationLoss,
    "podnet": PoolingDistillationLoss,
    "relation": RelativeDistancesLoss,
    "GradCAM": GradCAMLoss,
    "similitude": SimilitudeDistillationLoss,
    "GeoDL": GFK,
    "l1": L1Loss,
    "l2": L2Loss,
    "GeoDLxLUCIR": GeoDLxLUCIR,
    "always-be-dreaming": AlwaysBeDreaming,
}

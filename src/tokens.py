import faiss
import numpy as np
from sklearn.decomposition import PCA

K = 256


def reduce_dim(emb, dim):
    return PCA(n_components=dim, whiten=True).fit_transform(emb.astype(np.float32)).astype(np.float32)


def item_codes(emb, digits=32, pca=128, train_items=None, graph=None, graph_w=0.0):
    # RPG's tokenizer: whitened PCA, then OPQ + PQ (8 bits per digit) trained on items that occur in training
    x = reduce_dim(emb, pca) if pca else emb.astype(np.float32)
    xq = x
    if graph_w:
        unit = lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-6)
        xq = np.concatenate([unit(x), graph_w * unit(graph)], 1)
    index = faiss.index_factory(xq.shape[1], f"OPQ{digits},IVF1,PQ{digits}x8", faiss.METRIC_INNER_PRODUCT)
    index.train(xq if train_items is None else xq[train_items])
    index.add(xq)
    lists = faiss.extract_index_ivf(faiss.downcast_index(index.index)).invlists
    raw = faiss.rev_swig_ptr(lists.get_codes(0), lists.list_size(0) * lists.code_size)
    return raw.reshape(-1, lists.code_size).astype(np.int64), x

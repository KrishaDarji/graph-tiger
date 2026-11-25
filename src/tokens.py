import faiss
import numpy as np

K = 256


def reduce_dim(emb, dim):
    x = emb.astype(np.float32)
    m = faiss.PCAMatrix(x.shape[1], dim)
    m.train(x)
    return m.apply(x)


def item_codes(emb, digits=32, pca=128, graph=None, graph_w=0.0):
    # PCA-reduce the text vectors, then OPQ into `digits` codewords of 8 bits each
    x = reduce_dim(emb, pca) if pca else emb.astype(np.float32)
    xq = x
    if graph_w:
        unit = lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-6)
        xq = np.concatenate([unit(x), graph_w * unit(graph)], 1)
    opq = faiss.index_factory(xq.shape[1], f"OPQ{digits},PQ{digits}x8")
    opq.train(xq)
    opq.add(xq)
    codes = faiss.vector_to_array(faiss.downcast_index(opq.index).codes).reshape(len(x), digits).astype(np.int64)
    return codes, x

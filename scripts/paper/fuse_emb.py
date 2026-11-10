import sys
import numpy as np

content, graph, w, out = np.load(sys.argv[1]), np.load(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
unit = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
np.save(out, np.concatenate([unit(content), w * unit(graph)], 1).astype(np.float32))

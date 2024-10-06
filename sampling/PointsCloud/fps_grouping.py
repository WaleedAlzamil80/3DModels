from .FPS import FPS
from .Grouping import index_point, Grouping

def fbsGrouping(x, fea=None, args=None):
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    if not fea:
        fea = x

    centroids_idx = FPS(x, args.n_centroids)
    centroids = index_point(x, centroids_idx)
    x_points, g_points, labels, idx = Grouping(x, fea, centroids, args.nsamples, args.radius)
    return centroids, x_points, g_points, labels, idx

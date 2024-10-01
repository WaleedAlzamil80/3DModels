from .FPS import FPS
from .Grouping import index_point, Grouping

def fbsGrouping(x, n_centroids=1024, nsamples=32, radius=0.5):
    centroids_idx = FPS(x, n_centroids)
    centroids = index_point(x, centroids_idx)
    x_points, g_points, labels, idx = Grouping(x, x, centroids, nsamples, radius)
    return x_points.squeeze(0), idx

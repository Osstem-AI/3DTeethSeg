import numpy as np
import faiss

def distance_equation(point_1, point_2):
    distance = np.sqrt(np.sum((point_1-point_2) ** 2, axis=0))
    return distance

def ellipse_equation(x, y, radius):
        eli = (x*x)/((0.95*radius)*(0.95*radius)) + (y*y)/((1.2*radius)*(1.2*radius))
        return eli
    
class FaissKNeighbors(object):
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def _fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def _predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions
import numpy as np
import faiss

def extract_info(mesh, n_face=None):
    if n_face:
        mesh = mesh.simplify_quadratic_decimation(n_face)
        N = len(mesh.faces)
        points = np.array(mesh.vertices)
        cell_ids = np.array(mesh.faces)
        cells = points[cell_ids].reshape(N, 9).astype(dtype='float32')
        mean_center = np.array(mesh.vertices.sum(axis=0) / mesh.vertices.shape[0])
        mesh_normals = np.array(mesh.face_normals)
        barycenters = np.array(mesh.triangles_center)
        return cells, mean_center, mesh_normals, barycenters, cell_ids
    else:
        points = np.array(mesh.vertices)
        points_normals = np.array(mesh.vertex_normals)
        return points, points_normals
    
    
class CalcEdges:
    def __init__(self, adj_idx, normals, barycenters):
        self.normals = normals
        self.barycenters = barycenters
        self.adj_idx = adj_idx
        
    def _is_exists(self, it):
        return (it is not None)

    def _calc_theta(self, i_node):
        temp = []
        for i_nei in self.adj_idx[i_node]:
            cos_theta = np.dot(self.normals[i_node, 0:3], self.normals[i_nei, 0:3])/np.linalg.norm(
                self.normals[i_node, 0:3])/np.linalg.norm(self.normals[i_nei, 0:3])
            if cos_theta >= 1.0:
                cos_theta = 0.9999
            theta = np.arccos(cos_theta)
            phi = np.linalg.norm(self.barycenters[i_node, :] - self.barycenters[i_nei, :])
            if theta > np.pi/2.0:
                temp.append([i_node, i_nei, -np.log10(theta/np.pi)*phi])
            else:
                beta = 1 + np.linalg.norm(np.dot(self.normals[i_node, 0:3], self.normals[i_nei, 0:3]))
                temp.append([i_node, i_nei, -beta*np.log10(theta/np.pi)*phi])
            
        return np.array(temp) if len(temp) != 0 else None

    def __call__(self) -> (np.array):
        round_factor = 100
        lambda_c = 30
        edge_scores = list(filter(self._is_exists, map(self._calc_theta, range(0, len(self.adj_idx)))))
        edges = np.vstack(edge_scores[:])
        edges[:, 2] *= lambda_c*round_factor
        edges = edges.astype(np.int32)
        return edges
    
class FaissKNeighbors:
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
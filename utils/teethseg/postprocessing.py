import numpy as np
from configs import config
from pygco import cut_from_graph
from utils.teethseg.tools import FaissKNeighbors

class CalcEdges(object):
    def __init__(self, adj_idx, normals, barycenters):
        self.adj_idx = adj_idx
        self.normals = normals
        self.barycenters = barycenters
        
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
        edge_scores = list(filter(self._is_exists, map(self._calc_theta, range(0, len(self.adj_idx)))))
        edges = np.vstack(edge_scores[:])
        edges[:, 2] *= config['LAMBDA_C']*config['ROUND_FACTOR']
        edges = edges.astype(np.int32)
        return edges

class PostProcessing(object):
    def __init__(self):
        pass
    
    def _init_refinement(self, cropped_mesh_prob, newList_X):
        pairwise = (1 - np.eye(config['NUM_CLASSES'], dtype=np.int32))
        cropped_mesh_prob[cropped_mesh_prob<1.0e-6] = 1.0e-6
        
        # unaries
        unaries = -config['ROUND_FACTOR'] * np.log10(cropped_mesh_prob)
        unaries = unaries.astype(np.int32)
        unaries = unaries.reshape(-1, config['NUM_CLASSES'])

        adj_face_idxs = []
        selected_cell_ids = self.cell_ids[newList_X, :]
        for num_f in range(len(selected_cell_ids)):
            nei = np.sum(np.isin(selected_cell_ids, selected_cell_ids[num_f, :]), axis=1)
            nei_id = np.where(nei==2)
            nei_id = list(nei_id[0][np.where(nei_id[0] > num_f)])
            adj_face_idxs.append(sorted(nei_id))

        return adj_face_idxs, unaries, pairwise
    
    def _extract_mesh_info(self, mesh, n_face=None):
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
        
    def _refinement(self, cropped_mesh_prob, newList_X):
        adj_face_idxs, unaries, pairwise = self._init_refinement(cropped_mesh_prob, newList_X)
        # Calculate edges & conduct graph-cut
        calc_edges = CalcEdges(adj_face_idxs, self.mesh_normals[newList_X], self.barycenters[newList_X])
        edges = calc_edges()
        refined_cropped_labels = cut_from_graph(edges, unaries, pairwise)
        return refined_cropped_labels
    
    def _upsampling(self, k):
        neigh = FaissKNeighbors(k)
        neigh._fit(np.concatenate([self.barycenters, self.mesh_normals], axis=1), self.refine_labels)
        fine_labels_n = neigh._predict(np.concatenate([self.points_ori, self.points_normals_ori], axis=1))
        return fine_labels_n
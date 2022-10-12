import numpy as np
import torch
import math
import open3d as o3d
from configs import config
from utils.teethseg.preprocessing import model_loading
from utils.teethseg.mesh_processing import extract_info, CalcEdges, FaissKNeighbors
from utils.teethseg.utils import distance_equation, ellipse_equation
from pygco import cut_from_graph

class TeethSeg:
    def __init__(self, mesh, num_teeth, pts, device):
        self.mesh_original = mesh
        self.num_teeth = num_teeth
        self.pts = pts
        self.device = device
        self.checkpoint_path = './experiments'
        
    def _mesh_crop(self, n, barycenters, mean_cell_centers):
        barycenters_xy = barycenters[:, :2]
        dist = lambda input_p: min(list(map(lambda x: distance_equation(input_p, np.delete(self.pts, n, axis=0)[x, :2]), range(0, len(np.delete(self.pts, n, axis=0))))))
        radius = dist(self.pts[n, :2])
        
        if radius > config['THRES_RADIUS'][1]:
            radius = config['RADIUS_MEDIAN']
        elif radius < config['THRES_RADIUS'][0]:
            radius += 1
        
        if (self.num_teeth[n] >= 4) and (self.num_teeth[n] <= 13):
            moved_xy = barycenters_xy[:,:2] - self.pts[n, :2]
            mass_point_xy = mean_cell_centers[:2]
            radian = math.atan2(self.pts[n, :2][1] - mass_point_xy[1], self.pts[n, :2][0] - mass_point_xy[0])
            degree =  (radian * 180 / math.pi) + 90
            
            radian = np.radians(-degree)
            c, s = np.cos(radian), np.sin(radian)
            R = np.array(((c, -s), (s, c)))
            rotated_xy = np.dot(R, moved_xy.T).T
            
            distance_ellipse = ellipse_equation(rotated_xy[:,0], rotated_xy[:,1], radius)
            
            newList_X_temp = sorted(set(list(np.argwhere(distance_ellipse <= 1)[:,0])))
        else:
            distance_map_xy = np.array(list(map(lambda x: distance_equation(self.pts[n, :2], barycenters_xy[x]), range(0, len(barycenters_xy)))))
            newList_X_temp = sorted(set(list(np.argwhere(distance_map_xy < radius)[:,0])))
        
        # Remove outlier of Cropped mesh
        barycenters_teeth_temp = barycenters[newList_X_temp]
        line_set = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(barycenters_teeth_temp)
        )
        _, barycenters_ind = line_set.remove_statistical_outlier(nb_neighbors=40, std_ratio=4.0)
        newList_X = list(np.array(newList_X_temp)[barycenters_ind])
        
        return newList_X
    
    def _init_refinement(self, cropped_mesh_prob, cell_ids, newList_X):
        # refinement
        round_factor = config['ROUND_FACTOR']
        cropped_mesh_prob[cropped_mesh_prob<1.0e-6] = 1.0e-6

        # unaries
        unaries = -round_factor * np.log10(cropped_mesh_prob)
        unaries = unaries.astype(np.int32)
        unaries = unaries.reshape(-1, config['NUM_CLASSES'])

        # parawise
        pairwise = (1 - np.eye(config['NUM_CLASSES'], dtype=np.int32))
        
        adj_face_idxs = []
        selected_cell_ids = cell_ids[newList_X, :]
        for num_f in range(len(selected_cell_ids)):
            nei = np.sum(np.isin(selected_cell_ids, selected_cell_ids[num_f, :]), axis=1)
            nei_id = np.where(nei==2)
            nei_id = list(nei_id[0][np.where(nei_id[0] > num_f)])
            adj_face_idxs.append(sorted(nei_id))
            
        return adj_face_idxs, unaries, pairwise
        
        
    def __call__(self):
        points_ori, points_normals_ori = extract_info(self.mesh_original)
        cells, mean_cell_centers, mesh_normals, barycenters, cell_ids = extract_info(self.mesh_original, config['DECIMATION_FACTOR'])
        
        refine_labels = np.zeros(len(cell_ids), dtype=np.int32)
        
        models = model_loading(self.device)
        for i, t_id in enumerate(self.num_teeth):
            prediction_model = models[config['MODEL_CONFIG'][0][t_id]-1]
            with torch.no_grad():
                newList_X = self._mesh_crop(i, barycenters, mean_cell_centers)
                
                cells_teeth = cells[newList_X]
                barycenters_teeth = barycenters[newList_X]
                
                mean_teeth_centers = np.array(barycenters_teeth.sum(axis=0) / barycenters_teeth.shape[0])
                    
                cells_teeth[:, 0:3] -= mean_teeth_centers[0:3]
                cells_teeth[:, 3:6] -= mean_teeth_centers[0:3]
                cells_teeth[:, 6:9] -= mean_teeth_centers[0:3]
                
                barycenters_teeth[:, 0:3] -= mean_teeth_centers[0:3]
                normals_teeth = mesh_normals[newList_X]
                barycenters_norm = barycenters[newList_X]
                barycenters_norm -= mean_teeth_centers[0:3]
                
                maxs = barycenters_teeth.max(axis=0)
                mins = barycenters_teeth.min(axis=0)
                means = barycenters_teeth.mean(axis=0)
                stds = barycenters_teeth.std(axis=0)
                nmeans = normals_teeth.mean(axis=0)
                nstds = normals_teeth.std(axis=0)
                
                for i in range(3):
                    cells_teeth[:, i] = (cells_teeth[:, i] - means[i]) / stds[i] #point 1
                    cells_teeth[:, i+3] = (cells_teeth[:, i+3] - means[i]) / stds[i] #point 2
                    cells_teeth[:, i+6] = (cells_teeth[:, i+6] - means[i]) / stds[i] #point 3
                    barycenters_norm[:,i] = (barycenters_norm[:,i] - mins[i]) / (maxs[i]-mins[i])
                    normals_teeth[:,i] = (normals_teeth[:,i] - nmeans[i]) / nstds[i]
                
                X = np.column_stack((cells_teeth, barycenters_norm, normals_teeth))
                X = X.transpose(1, 0)
                X = X.reshape([1, X.shape[0], X.shape[1]])
                X = torch.from_numpy(X).to(self.device, dtype=torch.float)
                
                tensor_prob_output = prediction_model(X).to(self.device, dtype=torch.float)
                output_cropped_mesh_prob = tensor_prob_output.cpu().detach().numpy()
                
                # Initialize settings of refinement
                adj_face_idxs, unaries, pairwise = self._init_refinement(output_cropped_mesh_prob, cell_ids, newList_X)
                
                # Calculate edges & conduct graph-cut
                calcedges = CalcEdges(adj_face_idxs, mesh_normals[newList_X], barycenters[newList_X])
                edges = calcedges()
                refine_labels_temp = cut_from_graph(edges, unaries, pairwise)
                
                # Aggregate
                teeth_ids = list(np.array(newList_X)[np.argwhere(refine_labels_temp != 0)[:,0]])
                refine_labels[teeth_ids] = t_id
        
        neigh = FaissKNeighbors(k=3)
        neigh._fit(np.concatenate([barycenters, mesh_normals], axis=1), refine_labels)
        fine_labels_n = neigh._predict(np.concatenate([points_ori, points_normals_ori], axis=1))
    
        return fine_labels_n
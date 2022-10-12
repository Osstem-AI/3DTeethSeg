import os
import torch
import numpy as np
import open3d as o3d
import math
import lib.models.net as net
from configs import config
from utils.teethseg.tools import distance_equation, ellipse_equation

class PreProcessing(object):
    def __init__(self, mesh, device):
        self._models = self._model_loading(device)
        self._mesh_information(mesh)
        self.device = device
        
    def _model_loading(self, device):
        models = []
        for i in range(3):
            prediction_model = net.MeshSegNet(num_classes=config['NUM_CLASSES'], num_channels=15, with_dropout=True, dropout_p=0.5).to(device, dtype=torch.float)
            model_checkpoint_name = 'teeth_model'+str(i+1)+'.tar'
            model_checkpoint_path = os.path.join(config['CHECKPOINT_PATH'], model_checkpoint_name)
            checkpoint = torch.load(model_checkpoint_path, map_location=device)
            prediction_model.load_state_dict(checkpoint['model_state_dict'])
            prediction_model.eval()
            models.append(prediction_model)
        return models
    
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
    
    def _mesh_information(self, mesh):
        self.points_ori, self.points_normals_ori = self._extract_mesh_info(mesh)
        self.cells, self.mean_cell_centers, self.mesh_normals, self.barycenters, self.cell_ids = self._extract_mesh_info(mesh, config['DECIMATION_FACTOR'])
        
    def _remove_outliers(self, cropped_idx):
        # Remove outlier of Cropped mesh
        barycenters_teeth_temp = self.barycenters[cropped_idx]
        line_set = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(barycenters_teeth_temp)
        )
        _, barycenters_ind = line_set.remove_statistical_outlier(nb_neighbors=40, std_ratio=4.0)
        return barycenters_ind
    
    def _mesh_crop(self, n):
        barycenters_xy = self.barycenters[:, :2]
        dist = lambda input_p: min(list(map(lambda x: distance_equation(input_p, np.delete(self.pts, n, axis=0)[x, :2]), range(0, len(np.delete(self.pts, n, axis=0))))))
        radius = dist(self.pts[n, :2])
        
        if radius > config['THRES_RADIUS'][1]:
            radius = config['RADIUS_MEDIAN']
        elif radius < config['THRES_RADIUS'][0]:
            radius += 1
        
        if (self.num_teeth[n] >= 4) and (self.num_teeth[n] <= 13):
            moved_xy = barycenters_xy[:,:2] - self.pts[n, :2]
            mass_point_xy = self.mean_cell_centers[:2]
            radian = math.atan2(self.pts[n, :2][1] - mass_point_xy[1], self.pts[n, :2][0] - mass_point_xy[0])
            degree =  (radian * 180 / math.pi) + 90
            radian = np.radians(-degree)
            c, s = np.cos(radian), np.sin(radian)
            R = np.array(((c, -s), (s, c)))
            rotated_xy = np.dot(R, moved_xy.T).T
            distance_ellipse = ellipse_equation(rotated_xy[:,0], rotated_xy[:,1], radius)
            cropped_idx = sorted(set(list(np.argwhere(distance_ellipse <= 1)[:,0])))
        else:
            distance_map_xy = np.array(list(map(lambda x: distance_equation(self.pts[n, :2], barycenters_xy[x]), range(0, len(barycenters_xy)))))
            cropped_idx = sorted(set(list(np.argwhere(distance_map_xy < radius)[:,0])))
            
        removed_outliers_idx = self._remove_outliers(cropped_idx)
        newList_X = list(np.array(cropped_idx)[removed_outliers_idx])
        return newList_X
    
    def _normalize_mesh(self, newList_X):
        cells_teeth = self.cells[newList_X]
        barycenters_teeth = self.barycenters[newList_X]
        mean_teeth_centers = np.array(barycenters_teeth.sum(axis=0) / barycenters_teeth.shape[0])
            
        cells_teeth[:, 0:3] -= mean_teeth_centers[0:3]
        cells_teeth[:, 3:6] -= mean_teeth_centers[0:3]
        cells_teeth[:, 6:9] -= mean_teeth_centers[0:3]
        
        barycenters_teeth[:, 0:3] -= mean_teeth_centers[0:3]
        normals_teeth = self.mesh_normals[newList_X]
        
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
            barycenters_teeth[:,i] = (barycenters_teeth[:,i] - mins[i]) / (maxs[i]-mins[i])
            normals_teeth[:,i] = (normals_teeth[:,i] - nmeans[i]) / nstds[i]
            
        return cells_teeth, barycenters_teeth, normals_teeth
    
    def _generate_input(self, newList_X):
        cells_teeth, barycenters_teeth, normals_teeth = self._normalize_mesh(newList_X)
        X = np.column_stack((cells_teeth, barycenters_teeth, normals_teeth))
        X = X.transpose(1, 0)
        X = X.reshape([1, X.shape[0], X.shape[1]])
        X = torch.from_numpy(X).to(self.device, dtype=torch.float)
        return X
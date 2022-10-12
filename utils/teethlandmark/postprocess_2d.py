import numpy as np
from sklearn.cluster import DBSCAN


def optimize_depth_location(pred_3ds):
    return pred_3ds[np.argmax(pred_3ds[:, 2])]


class PostProcess2DLandmark(object):
    """Postprocessing mesh from 2d Landmark in captured image. project pixels(2d) to points(3d) in mesh

    Args:
        num_teeth (list):                                                                             Default: 16.
        score_map (list, BxCxHxW):
        predictions (str): 2d heatmaps prediction results                                             Default: 256x256.
        image_matrix (np.array): the list of pixels in captured image corresponding vertices index
        world_matrix (np.array): the list of points in mesh corresponding vertices index

    Returns:
        predictions_3d (list, xyz): 3d landmarks in mesh

    Attributes:
        num_teeth (np.array):
        score_map (tuple, x, y):
        predictions_2d(str)
        _world_matrix (np.array):
        _image_matrix(np.array):
        predictions_3d(int):
        _candidates(int):
        _outlier(int):
    """

    def __init__(self, num_teeth: list,
                 score_map: list,
                 predictions: list,
                 image_matrix: list,
                 world_matrix: np.array):

        self.num_teeth = num_teeth
        self.score_map = score_map
        self.predictions_2d = predictions
        self._world_matrix = world_matrix
        self._image_matrix = np.asarray(image_matrix)
        self.predictions_3d = []
        self._candidates = 30
        self._outlier = 40

    def __call__(self):
        self._cluster_by_dbscan()
        self._remove_outlier_vertices()
        self._sorting_by_mesh_depth()
        self.projection_2d_to_3d()
        return self.num_teeth, self.predictions_3d

    def _cluster_by_dbscan(self):

        #TODO: code refactorying
        clustering = DBSCAN(eps=10, min_samples=2).fit(self.predictions_2d)
        cluster_idx = np.unique(clustering.labels_)
        cluster_idx = np.delete(cluster_idx, np.where(cluster_idx == -1))

        if len(cluster_idx) != 0:
            for i in list(cluster_idx):
                core_idx = np.where(clustering.labels_ == i)[0]
                score_list = []
                for j in list(core_idx):
                    score_list.append(self.score_map[0][(self.num_teeth - 1)[j]].max())

                score_list_pop = score_list.copy()
                score_list_pop.remove(max(score_list))

                pts_idx = [core_idx[score_list.index(f)] for f in score_list_pop]

                for delete_idx in pts_idx:
                    self.predictions_2d[delete_idx, :] = -1
                    self.num_teeth[delete_idx] = -1

    def _remove_outlier_vertices(self):
        # TODO: code refactorying
        self.predictions_2d = self.predictions_2d[np.ravel(np.argwhere(self.predictions_2d[:, 0] + self.predictions_2d[:, 1] > self._outlier))]
        self.num_teeth = self.num_teeth[np.ravel(np.argwhere(self.predictions_2d[:, 0] + self.predictions_2d[:, 1] > self._outlier))]

    def _sorting_by_mesh_depth(self):
        sort_ix = np.argsort(self._world_matrix[..., -1])
        self._obj_pts_z_sort = self._world_matrix[sort_ix]
        self._img_pts_z_sort = self._image_matrix[sort_ix]

    def _calculate_minimum_distance(self, pred):
        # find the mininum distance index between predictions and screen(2d) matrix
        return np.linalg.norm(self._img_pts_z_sort[:, :2] - pred, axis=1)

    def projection_2d_to_3d(self):
        for num, pred in enumerate(self.predictions_2d):
            pred_3ds = self._obj_pts_z_sort[np.argsort(self._calculate_minimum_distance(pred))[:self._candidates]]
            self.predictions_3d.append(optimize_depth_location(pred_3ds))

        return self.predictions_3d
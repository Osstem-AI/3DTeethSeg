import trimesh
import numpy as np

class ToothMeshInfo(object):
    """Get tooth mesh information from stl file.

    Args:
        mesh (trimesh.Trimesh): get mesh from Trimesh API

    Attributes:
        _faces (list): the list of mesh faces using Trimesh API
        _points (np.array): the list of mesh points using Trimesh API
    """

    def __init__(self, mesh):
        self._mesh = mesh
        self._faces = np.asarray(self._mesh.faces).tolist()
        self._points = np.asarray(self._mesh.vertices)
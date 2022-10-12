import vtk
import numpy as np
import open3d as o3d
from vtk.util import numpy_support as np_support
from .info import ToothMeshInfo


class CaptureToothImage(ToothMeshInfo):
    """Capture tooth image from mesh data

    Args:
        mesh (openmesh.Trimesh): get mesh from Trimesh API.

    Attributes:
        _resolution (tuple, h x w): Input 2d image size. Default: (512, 512)
        _view (tuple, x, y, z): the view up direction for the camera. Default: (0.0, 1.0, 0.0).
        _zoom_factor (int): the view scales by the specified factor. Default: 30.
        image_matrix (list): the list of pixels in captured image corresponding vertices index.
        world_matrix (np.array): the list of points in mesh corresponding vertices index.
    """

    def __init__(self, mesh):
        super().__init__(mesh)
        self._resolution = (512, 512)
        self._view = (0.0, 1.0, 0.0)
        self._zoom_factor = 30
        self.image_matrix = []

    def __call__(self):
        self.catpure_screen_shot()
        self._setting_caputre()
        self._capture_tooth_image()
        self._get_world_to_image_matrix()
        return self.img, self.image_matrix, self.world_matrix
    
    def catpure_screen_shot(self):
        pcd_mesh = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(self._barycenters)))
        _, ind = pcd_mesh.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
        self._faces = np.asarray(self._faces)[ind]

    def _create_polygon(self):
        numberPoints = len(self._points)
        Points = vtk.vtkPoints()
        points_vtk = np_support.numpy_to_vtk(self._points, deep=1, array_type=vtk.VTK_FLOAT)
        Points.SetNumberOfPoints(numberPoints)
        Points.SetData(points_vtk)

        Triangles = vtk.vtkCellArray()

        for item in self._faces:
            Triangle = vtk.vtkTriangle()
            Triangle.GetPointIds().SetId(0, item[0])
            Triangle.GetPointIds().SetId(1, item[1])
            Triangle.GetPointIds().SetId(2, item[2])
            Triangles.InsertNextCell(Triangle)

        self._polydata = vtk.vtkPolyData()
        self._polydata.SetPoints(Points)
        self._polydata.SetPolys(Triangles)

        self._min_val, self._max_val = self._polydata.GetPoints().GetData().GetRange()
        self.world_matrix = np.asarray(self._polydata.GetPoints().GetData())

    def _set_look_up_table(self):
        # transfer function (lookup table) for mapping point scalar data to colors (parent class is vtkScalarsToColors)
        self._lut = vtk.vtkColorTransferFunction()
        self._lut.AddRGBPoint(self._min_val, 0.0, 0.0, 1.0)
        self._lut.AddRGBPoint(self._min_val + (self._max_val - self._min_val) / 4, 0.0, 0.5, 0.5)
        self._lut.AddRGBPoint(self._min_val + (self._max_val - self._min_val) / 2, 0.0, 1.0, 0.0)
        self._lut.AddRGBPoint(self._min_val - (self._max_val - self._min_val) / 4, 0.5, 0.5, 0.0)
        self._lut.AddRGBPoint(self._min_val, 1.0, 0.0, 0.0)

    def _set_mapper(self):
        self._mapper = vtk.vtkPolyDataMapper()
        self._mapper.SetLookupTable(self._lut)
        self._mapper.SetScalarRange(self._min_val, self._max_val)
        self._mapper.SetInputData(self._polydata)

    def _set_actor(self):
        self._actor = vtk.vtkActor()
        self._actor.SetMapper(self._mapper)

    def _set_render_window(self, ren=None):
        if ren is None:
            self._ren = vtk.vtkRenderer()
        else:
            self._ren = ren

        self._renWin = vtk.vtkRenderWindow()
        self._renWin.SetOffScreenRendering(1)
        self._renWin.AddRenderer(self._ren)
        self._renWin.SetSize(self._resolution)
        self._ren.SetBackground(0, 0, 0)

    def _set_interact_rendering_with_camera(self):
        # create a renderwindowinteractor
        self._iren = vtk.vtkRenderWindowInteractor()
        self._iren.SetRenderWindow(self._renWin)
        self._ren.AddActor(self._actor)
        self._renWin.Render()

    def _set_camera(self):
        self._set_render_window()
        self._set_interact_rendering_with_camera()
        # Renderer (Zoom in)
        pos = self._ren.GetActiveCamera().GetPosition()
        foc = self._ren.GetActiveCamera().GetFocalPoint()

        # Re-Renderer (Zoom in)
        self._ren = vtk.vtkRenderer()
        self._camera = vtk.vtkCamera()
        self._camera.SetViewUp(self._view)
        self._camera.SetPosition(pos[0], pos[1], pos[2] - self._zoom_factor)
        self._camera.SetFocalPoint(foc[0], foc[1], foc[2])
        self._ren.SetActiveCamera(self._camera)

        self._set_render_window(self._ren)
        self._set_interact_rendering_with_camera()

    def _set_image_filter(self):
        self._renWin.Render()
        self._grabber = vtk.vtkWindowToImageFilter()
        self._grabber.SetInput(self._renWin)
        self._grabber.Update()

    def _setting_caputre(self):
        self._create_polygon()
        self._set_look_up_table()
        self._set_mapper()
        self._set_actor()
        self._set_camera()
        self._set_image_filter()

    def _capture_tooth_image(self):
        self.img = np.asarray(self._grabber.GetOutput().GetPointData().GetScalars())
        self.img = self.img.reshape(self._resolution + (3, ))

    def _get_world_to_image_matrix(self) -> np.array:
        for p in self.world_matrix:
            displayPt = [0, 0, 0]
            vtk.vtkInteractorObserver.ComputeWorldToDisplay(self._ren, p[0], p[1], p[2], displayPt)
            self.image_matrix.append(displayPt)

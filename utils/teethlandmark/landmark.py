import torch
import numpy as np
import lib.models.net as net
from lib.config import config_hrnet
from lib.core.evaluation import decode_preds


def remove_outlier_predictions(pts):
    # Remove outlier(Null) points in the top-left
    return np.argwhere(pts[:, 0] != 0)[:, 0]


class Predict2DToothLandmark(object):
    """Predict 2d Landmarks in captured image.

    Args:
        img (np.array): Captured Image.                                      Default: 256x256.
        device (int): CPU/GPU mode define by device                          Default:CPU
        model_path (str): path to a model file                               Default:'./experiments/landmark_upper.pth'

    Returns:
        landmarks (list): 2D landmarks

    Attributes:
        _img (np.array): Captured Image.                                     Default: 256x256.
        _device (str): CPU/GPU mode define by device                         Default:CPU
        _model_path (str): 2d landmark model path
        _output_size (list, h x w): input 2d image size                      Default: 128x128.
        _interpolation_ratio (int): the input size to decode original size   Default: 4
    """

    def __init__(self, img: np.array, device: str, model_path: str):
        self._img = img
        self._device = device
        self._model_path = model_path
        self._output_size = [128, 128]
        self._interpolation_ratio = 4

    def __call__(self):
        self._make_transforms_from_screenshot()
        self._loading_2d_landmark_model()
        self._predict_2d_heatmaps()
        self._decode_heatmaps_to_landmarks()
        return self.pred_teeth, self.landmarks, self.output

    def _make_transforms_from_screenshot(self):
        self._input = self._img.astype(np.float32) / 255.0
        self._input = self._input.transpose([2, 0, 1])
        self._input = torch.from_numpy(self._input)

        if self._device == 'cuda':
            self._input = self._input.cuda()
        self._input.unsqueeze_(0)

    def _loading_2d_landmark_model(self):
        weight = torch.load(self._model_path, map_location=self._device)
        self._model = net.get_x_lay_landmark_net(config_hrnet)
        if self._device == 'cuda':
            self._model.cuda()

        try:
            self._model.load_state_dict(weight['state_dict'].state_dict())
        except:
            self._model.load_state_dict(weight)

        self._model.eval()

    @torch.no_grad()
    def _predict_2d_heatmaps(self):
        self.output = self._model(self._input).data.cpu()

    def _decode_heatmaps_to_landmarks(self):
        preds = decode_preds(self.output, self._output_size)
        pts = np.asarray(preds[0] * self._interpolation_ratio)
        self.pred_teeth = remove_outlier_predictions(pts)
        self.landmarks = pts[self.pred_teeth]
        self.pred_teeth += 1




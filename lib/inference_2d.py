import os
from configs import config
from utils.teethlandmark.capture import CaptureToothImage
from utils.teethlandmark.landmark import Predict2DToothLandmark
from utils.teethlandmark.postprocess_2d import PostProcess2DLandmark

def refined_seg_2d(mesh, jaw, device):
    file_name = 'landmark_{}.pth'.format(jaw)
    model_path = os.path.join(config['CHECKPOINT_PATH'], file_name)
    img, projection_2d, projection_3d = CaptureToothImage(mesh)()
    num_teeth, landmarks, score_map = Predict2DToothLandmark(img, device=device, model_path=model_path)()
    num_teeth, pred = PostProcess2DLandmark(num_teeth, score_map, landmarks, projection_2d, projection_3d)()

    return pred, num_teeth





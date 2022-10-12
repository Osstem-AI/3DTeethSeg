import numpy as np
import torch
from configs import config
from utils.teethseg.preprocessing import PreProcessing
from utils.teethseg.postprocessing import PostProcessing

class TeethSeg(PreProcessing, PostProcessing):
    def __init__(self, mesh, num_teeth, pts, device):
        PreProcessing.__init__(self, mesh, device)
        self.refine_labels = np.zeros(len(self.cell_ids), dtype=np.int32)
        self.num_teeth = num_teeth
        self.pts = pts
        
    def _inference_teeth(self):
        for i, t_id in enumerate(self.num_teeth):
            prediction_model = self._models[config['MODEL_CONFIG'][0][t_id]-1]
            with torch.no_grad():
                newList_X = PreProcessing._mesh_crop(self, i)
                X = PreProcessing._generate_input(self, newList_X)
                
                tensor_prob_output = prediction_model(X).to(self.device, dtype=torch.float)
                output_cropped_mesh_prob = tensor_prob_output.cpu().detach().numpy()
                
                # Refinement
                refined_cropped_labels = PostProcessing._refinement(self, output_cropped_mesh_prob, newList_X)
                
                # Aggregate
                teeth_ids = list(np.array(newList_X)[np.argwhere(refined_cropped_labels != 0)[:,0]])
                self.refine_labels[teeth_ids] = t_id
        
    def __call__(self):
        self._inference_teeth()
        upsampled_labels = PostProcessing._upsampling(self, k=3)
        
        return upsampled_labels
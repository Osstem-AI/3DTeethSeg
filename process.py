import glob
import os
import numpy as np
import torch
import json
import traceback
import trimesh
from lib.teeth_seg import TeethSeg
from lib.inference_2d import refined_seg_2d
from configs import config

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class ScanSegmentation():  # SegmentationAlgorithm is not inherited in this class anymore
    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        pass

    @staticmethod
    def load_input(input_dir):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """

        # iterate over files in input_dir, assuming only 1 file available
        inputs = glob.glob(f'{input_dir}/*.obj')
        print("scan to process:", inputs)
        return inputs

    @staticmethod
    def write_output(labels, instances, jaw):
        """
        Write to /output/dental-labels.json your predicted labels and instances
        Check https://grand-challenge.org/components/interfaces/outputs/
        """
        pred_output = {'id_patient': "",
                       'jaw': jaw,
                       'labels': labels,
                       'instances': instances
                       }

        # just for testing
        with open('./output/dental-labels.json', 'w') as fp:
            json.dump(pred_output, fp, cls=NpEncoder)

        return

    @staticmethod
    def get_jaw(scan_path):
        try:
            # read jaw from filename
            _, jaw = os.path.basename(scan_path).split('.')[0].split('_')
        except:
            # read from first line in obj file
            try:
                with open(scan_path, 'r') as f:
                    jaw = f.readline()[2:-1]
                if jaw not in ["upper", "lower"]:
                    return None
            except Exception as e:
                print(str(e))
                print(traceback.format_exc())
                return None
        return jaw

    def model(self, mesh, device, jaw=None):
        xyz, pred_teeth = refined_seg_2d(mesh, jaw, device)
        teeth_seg = TeethSeg(mesh, pred_teeth.tolist(), np.vstack(xyz), device)
        instances = teeth_seg()
        
        fdi_system = config['LOWER_LABEL'][0] if jaw == 'lower' else config['UPPER_LABEL'][0]
        labels = [fdi_system[t] for t in instances]
        
        return labels, instances

    def predict(self, inputs, device):
        """
        Your algorithm goes here
        """

        try:
            assert len(inputs) == 1, f"Expected only one path in inputs, got {len(inputs)}"
        except AssertionError as e:
            raise Exception(e.args)
        scan_path = inputs[0]
        print(f"loading scan : {scan_path}")
        # read input 3D scan .obj
        try:
            # you can use trimesh or other any loader we keep the same order
            mesh = trimesh.load(scan_path, process=False)
            jaw = self.get_jaw(scan_path)
            print("jaw processed is:", jaw)
        except Exception as e:
            print(str(e))
            print(traceback.format_exc())
            raise
        # inference data here
        # extract number of vertices from mesh
        labels, instances = self.model(mesh, device, jaw=jaw)

        try:
            assert (len(labels) == len(instances) and len(labels) == mesh.vertices.shape[0]), \
                "length of output labels and output instances should be equal"
        except AssertionError as e:
            raise Exception(e.args)

        return labels, instances, jaw

    def process(self):
        """
        Read input from /input, process with your algorithm and write to /output
        assumption /input contains only 1 file
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        input = self.load_input(input_dir='./input')
        labels, instances, jaw = self.predict(input, device)
        self.write_output(labels=labels, instances=instances, jaw=jaw)

if __name__ == "__main__":
    ScanSegmentation().process()
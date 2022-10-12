import torch
import os
import lib.models.net as net
from configs import config

def model_loading(device):
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
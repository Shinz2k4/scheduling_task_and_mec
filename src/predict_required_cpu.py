import joblib
from trainer_compute_delay import DelayPredictor
import torch 

class CPUPredictor:
    def __init__(self,task):
        self.scaler = joblib.load(f"train_result/scaler_delay.pkl")
        self.model = DelayPredictor()
        self.model.load_model(f"train_result/pretrained_processing_estimation_compute_delay.pth")
        self.task = task
    def predict_required_cpu(self):
        task_scaled = self.scaler.transform(self.task)
        task_tensor = torch.tensor(task_scaled, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            compute_delay = self.model(task_tensor).item()
        cpu_required = compute_delay*3.1*10**9
        return cpu_required

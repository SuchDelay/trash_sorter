from torchvision import transforms
import torch
from PIL import Image
from network import Net  # your existing network definition

# Helper to move tensors to device
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class MyModel:
    def __init__(self, trained_weights: str, device: str):
        # Initialize model
        self.net = Net()
        self.weights = trained_weights
        self.device = torch.device('cuda:0' if device == 'cuda' else 'cpu')
        self._initialize()

    def _initialize(self):
        # Load model weights safely for CPU/GPU
        try:
            map_location = lambda storage, loc: storage  # CPU fallback
            self.net.load_state_dict(torch.load(self.weights, map_location=map_location)["state_dict"])
        except IOError:
            print("Error loading weights")
            return
        self.net.eval()
        self.net.to(self.device)

    def infer(self, path):
        # Open image and force 3 channels (RGB)
        img = Image.open(path).convert("RGB")

        # Preprocess image
        preprocess = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        image_tensor = preprocess(img)

        # Add batch dimension and move to device
        input_batch = to_device(image_tensor.unsqueeze(0), self.device)

        # Run inference
        with torch.no_grad():
            output = self.net(input_batch)

        # Get predicted class and confidence
        confidence, index = torch.max(output, dim=1)
        return index[0].item(), confidence[0].item()

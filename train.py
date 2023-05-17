import os
import torch
import torchvision.transforms as transforms
import pickle
from PIL import Image
from tqdm import tqdm
import pickle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class PretrainModel:
    
    def __init__(self):
        self.model = torch.load("models/pretrained.pkl")
        self.model.eval()
        self.transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def get_feature(self, pil_image):
        image_tensor = self.transformer(pil_image)
        image_tensor = image_tensor.unsqueeze(0)
        with torch.no_grad():
            feature = self.model(image_tensor).squeeze().detach().cpu().numpy()
        return feature

def read_training_data_label():
    with open("data/image_label.pkl", "rb") as f:
        return pickle.load(f)

def read_training_data():
    features_dict = {}
    folder_path = "data/images"
    pretrained = PretrainModel()
    for filename in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        pil_image = Image.open(img_path).convert("RGB")
        features_dict[filename] = pretrained.get_feature(pil_image)
        
    return features_dict

def create_model(name):
    X = np.array(list(read_training_data().values()))
    y = np.array(list(read_training_data_label().values()))
    knn = KNeighborsClassifier(n_neighbors = 10, p = 2, weights = 'uniform')
    knn.fit(X,y)
    with open(str(name) + ".pkl", "wb") as f:
        pickle.dump(knn, f)

# create_model("trained_model")
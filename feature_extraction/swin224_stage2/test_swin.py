import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from model_infer import Model, set_bn_eval

from albumentations.pytorch import ToTensorV2

import albumentations as A

import time



transform =  A.Compose([A.Resize(224, 224),
                                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                            max_pixel_value=255.0, p=1.0), ToTensorV2()], p=1.)

# # GPU
# device = torch.device("cuda:0")
# print(device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# CPU
# device="cpu"

model = Model("effb7", "SM", 512, num_classes=6000)
model.apply(set_bn_eval)
model.load_state_dict(torch.load('./model/swin224_stage2.pth'),
                      strict=False)

## time check
start = time.time()

img1_path = 'C:/Users/user/pet_bio/pet-biometrics/b7ns_stage2/demo_img/xPWXoOz0RJ2uGy381aUq9wAAACMAARAD.jpg'
img2_path = 'C:/Users/user/pet_bio/pet-biometrics/b7ns_stage2/demo_img/zusOFo_RSsqRAqtrfJ1fEgAAACMAARAD.jpg'

img1 = cv2.imdecode(np.fromfile(img1_path, dtype=np.uint8), 1)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# print(img1.shape)

img2 = cv2.imdecode(np.fromfile(img2_path, dtype=np.uint8), 1)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

trans1_img = transform(image=img1)['image']
trans2_img = transform(image=img2)['image']

# print(trans1_img.shape)
# trans_img = torch.unsqueeze(trans_img, 0)
# print(trans_img.unsqueeze(0).shape)
model.to(device)
model.eval()
# print(model())
with torch.no_grad():
    feature1 = model(trans1_img.unsqueeze(0).to(device))
    feature2 = model(trans2_img.unsqueeze(0).to(device))

    similarity = torch.cosine_similarity(feature1[0], feature2[0], dim=0)
    print(similarity)

# time check
end = time.time()

print(end - start)

print(feature1[0].shape)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from fastapi import FastAPI, File, UploadFile, Form
import torchvision.transforms as tf
import argparse
import os
import sys
import math
import pickle
from PIL import Image
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import base64
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import os
from io import BytesIO
from pydantic import BaseModel

FACENET_MODEL_PATH = './models/model.pt'
mtcnn = MTCNN(keep_all=True, post_process=False)
#get from config
num_classes = 4
class_name = ['jolie', 'john', 'badd', 'tri']
class_dict = dict([(i, class_name[i]) for i in range(num_classes)])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def get_face(image):
  mtcnn = MTCNN(keep_all=True, post_process=False)
  face = mtcnn(image)[0]
  image = transforms.Resize((160, 160))((transforms.ToTensor()((Image.fromarray(face.permute(1, 2, 0).numpy().astype(np.uint8))))))
  image = Image.fromarray(face.permute(1, 2, 0).numpy().astype(np.uint8))
  return image



# Load the model
print('Loading feature extraction model')
facenet = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=num_classes
).to(device)

facenet.load_state_dict(torch.load(FACENET_MODEL_PATH))
facenet.eval()

# Get input and output tensors by POST



app = FastAPI()

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

class Item(BaseModel):
    name: str
    description: str 
    owner: str
@app.post("/predict/")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    og_image = read_imagefile(await file.read())
    image = get_face(og_image)
    image = tf.ToTensor()(image)
    logits = facenet(image.unsqueeze(0))
    prob = torch.max(torch.nn.functional.softmax(logits)).item()
    name = class_dict[torch.argmax(logits).item()]
    return {'class': name, 'prob': prob}
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
from glob import glob
from dbController import dbController

WORKING_PATH = os.environ['dir']
MODEL_PATH = WORKING_PATH + '/source/models/current_model.pt'
DATABASE_PATH = WORKING_PATH + '/database'
data_dir = DATABASE_PATH + '/test_images'


#init Config for trained models
paths = glob(DATABASE_PATH + '/test_images/*')
num_classes = len(paths)
print("NUM CLASSES: ", num_classes)
class_name = [path.split('/')[-1] for path in paths]
class_dict = dict([(i, class_name[i]) for i in range(num_classes)])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
controller = dbController(num_classes, class_name)


#database QC
register_status = True
if len(glob(DATABASE_PATH + "*")) < num_classes:
    print("Checking the registration please")
    register_status = False
def get_face(image):
  mtcnn = MTCNN(keep_all=True, post_process=False)
  face = mtcnn(image)[0]
  image = transforms.Resize((160, 160))((transforms.ToTensor()((Image.fromarray(face.permute(1, 2, 0).numpy().astype(np.uint8))))))
  image = Image.fromarray(face.permute(1, 2, 0).numpy().astype(np.uint8))
  return image


#Model
print('Loading feature extraction model')
try:
    facenet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=num_classes
    ).to(device)
    facenet.load_state_dict(torch.load(MODEL_PATH))
    facenet.eval()
except:
    facenet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=num_classes - 1
    ).to(device)
    facenet.load_state_dict(torch.load(MODEL_PATH))
    facenet.eval()





#API
app = FastAPI()
def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file)) #binary file object
    return image
@app.post("/predict/")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    og_image = read_imagefile(await file.read()) #async so have to wait
    image = get_face(og_image)
    image = tf.ToTensor()(image)
    logits = facenet(image.unsqueeze(0))
    prob = torch.max(torch.nn.functional.softmax(logits, dim=-1)).item()
    name = class_dict[torch.argmax(logits).item()]
    # if prob < 0.7:
    #     name = "unknown"
    return {'class': name, 'prob': prob}

@app.post("/register/")
async def register_api(file1: UploadFile = File(...), file2: UploadFile = File(...), file3: UploadFile = File(...), file4: UploadFile = File(...), file5: UploadFile = File(...), name: str = Form(...), id: str = Form(...)):
    files = [file1, file2, file3, file4, file5]
    images = []
    for file in files:
        extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
        if not extension:
            return "Image must be jpg or png format!"
        image = BytesIO(await file.read())
        images.append(image)
    #Label
    controller.addRegistration(images, name, id)
    #Retrain
    controller.fit()

    return "Successfully registered"
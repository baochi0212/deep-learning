from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import fuckit
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
data_cropped = data_dir + '_cropped'
bins_dir = DATABASE_PATH + '/bins'
bin_list = [os.path.basename(i)[:-4] for i in glob(bins_dir + '/*.txt')]


#init Config for trained models
paths = glob(DATABASE_PATH + '/test_images/*')
num_classes = len(paths)
print("NUM CLASSES: ", num_classes)
class_name = [path.split('/')[-1] for path in paths]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
datasets = datasets.ImageFolder(data_cropped)
class_dict = dict((value, key) for key, value in datasets.class_to_idx.items())
#database QC
controller = dbController(num_classes, class_name)

# register_status = True
# if len(glob(DATABASE_PATH + "*")) < num_classes:
#     print("Checking the registration please")
#     register_status = False


def get_face(image):
    mtcnn = MTCNN(keep_all=True, post_process=False)
    face = mtcnn(image)[0]
    image = transforms.Resize((160, 160))((transforms.ToTensor()((Image.fromarray(face.permute(1, 2, 0).numpy().astype(np.uint8))))))
    image = Image.fromarray(face.permute(1, 2, 0).numpy().astype(np.uint8))
    return image
#module to skip the exception XD
def callFacenet(num_classes):
    facenet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=num_classes
    ).to(device)
    facenet.load_state_dict(torch.load(MODEL_PATH))
    return facenet
@fuckit
def modelLoading():
    facenet = callFacenet(num_classes)
    facenet = callFacenet(num_classes-1)
    facenet = callFacenet(num_classes+1)
    

    return facenet






#API
app = FastAPI()
def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file)) #binary file object
    return image
@app.post("/predict/")
async def predict_api(file: UploadFile = File(...), id: UploadFile = Form(...)):

    facenet = modelLoading()

    facenet.eval()
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    og_image = read_imagefile(await file.read()) #async so have to wait
    image = get_face(og_image)
    image = tf.ToTensor()(image)
    logits = facenet(image.unsqueeze(0))
    prob = torch.max(torch.nn.functional.softmax(logits, dim=-1)).item()
    idx = torch.argmax(logits).item()
    print("logits", logits)
    print("class name", class_name)
    name = class_dict[idx] if idx < num_classes else "unknown"
    if name in bin_list:
        name = "unknown"
    # if prob < 0.5:
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
    overlap = controller.addRegistration(images, name, id)
    #Retrain
    if overlap:
        controller.fit()
        return "Successfully registered"
    else:
        return "Existed"
    

@app.post("/delete/")
async def delete_api(id: str = Form(...), name: str = Form(...)):
    exist = controller.deleteRegister(name, id)
    if not exist:
        # controller.fit() for retrain if remove members
        return "Sucessfully deleted"
    else:
        return "Non-existed"

#if wanna programmatically run
# if __name__ == "__main__":
#     uvicorn.run("my_fastapi:app", port=8000, reload=True, access_log=False, reload_dirs=["/home/xps/projects/deep-learning-/CV/apps/Recognition/FaceVerify/deploy/database/test_images"])


# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# from fastapi import FastAPI, File, UploadFile, Form
# import uvicorn
# import fuckit
# import torchvision.transforms as tf
# import argparse
# import os
# import sys
# import math
# import pickle
# from PIL import Image
# import numpy as np
# import cv2
# import collections
# from sklearn.svm import SVC
# import base64
# from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
# import torch
# from torch.utils.data import DataLoader, SubsetRandomSampler
# from torch import optim
# from torch.optim.lr_scheduler import MultiStepLR
# from torchvision import datasets, transforms
# import numpy as np
# import pandas as pd
# import os
# from io import BytesIO
# from pydantic import BaseModel
# from glob import glob
# from dbController import dbController
# import pyodbc
# import cv2 as cv
# WORKING_PATH = os.environ['dir']
# MODEL_PATH = WORKING_PATH + '/source/models/current_model.pt'
# DATABASE_PATH = WORKING_PATH + '/database'
# data_dir = DATABASE_PATH + '/test_images'

# # conx = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server}; SERVER= DESKTOP-9IVV7T5\MSSQLSERVER01; Database=RFID_RECOG;UID=sa;PWD=12345')
# # cursor = conx.cursor()

# #init Config for trained models
# paths = glob(DATABASE_PATH + '/test_images/*')
# num_classes = len(paths)
# print("NUM CLASSES: ", num_classes)
# class_name = [path.split('/')[-1] for path in paths]
# class_dict = dict([(i, class_name[i]) for i in range(num_classes)]) 
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# controller = dbController(num_classes, class_name)


# #database QC
# register_status = True
# if len(glob(DATABASE_PATH + "*")) < num_classes:
#     print("Checking the registration please")
#     register_status = False
# def get_face(image):
#     mtcnn = MTCNN(keep_all=True, post_process=False)
#     face = mtcnn(image)[0]
#     image = transforms.Resize((160, 160))((transforms.ToTensor()((Image.fromarray(face.permute(1, 2, 0).numpy().astype(np.uint8))))))
#     image = Image.fromarray(face.permute(1, 2, 0).numpy().astype(np.uint8))
#     return image
# #module to skip the exception XD
# def callFacenet(num_classes):
#     facenet = InceptionResnetV1(
#         classify=True,
#         pretrained='vggface2',
#         num_classes=num_classes
#     ).to(device)
#     facenet.load_state_dict(torch.load(MODEL_PATH))
#     return facenet
# @fuckit
# def modelLoading():
#     facenet = callFacenet(num_classes)
#     facenet = callFacenet(num_classes-1)
#     facenet = callFacenet(num_classes+1)
    

#     return facenet






# #API
# app = FastAPI()
# def read_imagefile(file) -> Image.Image:
#     image = Image.open(BytesIO(file)) #binary file object
#     return image
# @app.post("/predict/")
# async def predict_api(file: UploadFile = File(...), id: str = Form(...)):

#     facenet = modelLoading()

#     facenet.eval()
#     extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
#     if not extension:
#         return "Image must be jpg or png format!"
#     og_image = read_imagefile(await file.read()) #async so have to wait
#     image = get_face(og_image)
#     image = tf.ToTensor()(image)
#     logits = facenet(image.unsqueeze(0))
#     prob = torch.max(torch.nn.functional.softmax(logits, dim=-1)).item()
#     idx = torch.argmax(logits).item()
#     name = class_dict[idx] if idx < num_classes else "unknown"
#     # if prob < 0.5:
#     #     name = "unknown"

#     # qry = "select * from USERS where RFID =" + id
#     #
#     # for row in cursor.execute(qry):
#     #     name1 = row.Name + "_" + id

#     #if name == name1:
#     return {'class': name, 'prob': prob}
#     #return "name: "+name + " || " +"pro: "+str(prob)+ " || name1: "+name1

# @app.post("/register/")
# async def register_api(file1: UploadFile = File(...), file2: UploadFile = File(...), file3: UploadFile = File(...), file4: UploadFile = File(...), file5: UploadFile = File(...), name: str = Form(...), id: str = Form(...)):
#     files = [file1, file2, file3, file4, file5]
#     images = []
#     for file in files:
#         extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
#         if not extension:
#             return "Chỉ chọn định dạng jpg || jpeg || png !!!"
#         image = BytesIO(await file.read())
#         images.append(image)
#     #Label
#     overlap = controller.addRegistration(images, name, id)
#     #Retrain
#     if overlap:
#         controller.fit()

#     return "Đăng ký thành công"

# @app.post("/getCamStatus/")
# async def getCamStatus(type: str = Form(...), link: str = Form(...)):
#     if type == "webcam":
#         cap = cv.VideoCapture(0)
#         if cap is None or not cap.isOpened():
#             return "fail"
#     else:
#         cap = cv.VideoCapture(link)
#         if cap is None or not cap.isOpened():
#             return "fail"

#     return "success"
    #return "name: "+name + " || " +"pro: "+str(prob)+ " || name1: "+name1
#if wanna programmatically run
# if __name__ == "__main__":
#     uvicorn.run("my_fastapi:app", port=8000, reload=True, access_log=False, reload_dirs=["/home/xps/projects/deep-learning-/CV/apps/Recognition/FaceVerify/deploy/database/test_images"])
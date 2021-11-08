import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import json
import requests
import glob

from collections import Counter

from matplotlib.figure import Figure
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from PIL import Image


#Visual_Arts2

def average_DH_person(imList):
    ims = []
    for im in imList:
        ims.append(Image.open(im, mode='r'))
    ims = np.array([np.array(im) for im in ims])
    avg = np.average(ims,axis=0)
    return Image.fromarray(avg.astype('uint8'))

def rgb_max(img):
    temp = img.copy()
    colgogh = temp.reshape(-1,3)
    unique, counts = np.unique(colgogh, axis=0, return_counts=True)
    r, g, b  = unique[np.argmax(counts)]
    return r, g, b


def rgb2hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def colour_preferences(images):
    cols = []
    for image in images:
        with open(image, 'rb') as file:
            img = Image.open(file)
            img = np.array(img)
            r, g, b = rgb_max(img)
            color = rgb2hex(r,g,b)
            cols.append(color)
    return cols


toby_url = 'https://www.kcl.ac.uk/newimages/ah/digital-humanities/people/blanketobias2.xc0244520.png'

def analyse_face(img_url= toby_url):
        
    ENDPOINT = 'ENDPOINT'
    face_api_url = ENDPOINT + '/face/v1.0/detect/'
    subscription_key = 'KEY'
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}
    params = {
        'returnFaceAttributes': 'age,gender,facialHair,emotion',
        'recognitionModel': 'recognition_01',
        'detectionModel': 'detection_01',
        'returnFaceId': 'true'
    }
    response = requests.post(face_api_url, params=params,
                         headers=headers, json={"url": img_url})
    d = json.loads(json.dumps(response.json()))
    return d[0]['faceAttributes'] 

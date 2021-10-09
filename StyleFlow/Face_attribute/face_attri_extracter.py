from io import BytesIO
import os
from PIL import Image, ImageDraw
import requests
import numpy as np

from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import FaceAttributeType
from msrest.authentication import CognitiveServicesCredentials

'''
This example detects faces from 2 different images, then returns information about their facial features.
The features can be set to a variety of properties, see the SDK for all available options.    
Prequisites:
Install the Face SDK: pip install --upgrade azure-cognitiveservices-vision-face
References:
Face SDK: https://docs.microsoft.com/en-us/python/api/azure-cognitiveservices-vision-face/?view=azure-python
Face documentation: https://docs.microsoft.com/en-us/azure/cognitive-services/face/
Face API: https://docs.microsoft.com/en-us/azure/cognitive-services/face/apireference
'''

'''
Authenticate the Face service
'''
# This key will serve all examples in this document.
KEY = ''

# This endpoint will be used in all examples in this quickstart.
ENDPOINT = ''

# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))


'''
Detect face(s) with attributes in a URL image
'''

# List of images PATH
images_list = sorted(os.listdir('images/'))

# Attributes you want returned with the API call, a list of FaceAttributeType enum (string format)
face_attributes = ['age', 'gender', 'headPose', 'smile', 'facialHair', 'glasses', 'emotion','hair']
attri_ids = []

# Detect a face with attributes, returns a list[DetectedFace]
for image in images_list:
    print(image)
    detected_faces = face_client.face.detect_with_stream(open('images/'+image,'rb'), return_face_attributes=face_attributes)
    if not detected_faces:
        raise Exception(
            'No face detected from image {}'.format(os.path.basename(image)))
    if len(detected_faces) > 1:
        detected_faces=[detected_faces[0]]
        print('Too many detected face!!! But choose one')

    # Face IDs are used for comparison to faces (their IDs) detected in other images.
    for face in detected_faces:
        """
        순서는 UI위치와 동일하게 놓는다.
        face.face_attributes.gender : 'female:0','male:1'
        face.face_attributes.glasses : 'noGlasses:0','readingGlasses:1'
        face.face_attributes.head_pose.yaw : 그대로
        face.face_attributes.head_pose.pitch : 그대로
        face.face_attributes.hair.bald : 그대로
        face.face_attributes.facial_hair.beard : 그대로
        face.face_attributes.age : 그대로
        face.face_attributes.smile : 그대로
        """
        attributes = [[0 if face.face_attributes.gender == 'female' else 1], [0 if face.face_attributes.glasses == 'noGlasses' else 1], [face.face_attributes.head_pose.yaw], \
            [face.face_attributes.head_pose.pitch], [face.face_attributes.hair.bald], [face.face_attributes.facial_hair.beard], [face.face_attributes.age], [face.face_attributes.smile]]

        attri_ids.append(attributes)
np.save('results/attributes.npy',np.float64(attri_ids))
print("JOB END!!")
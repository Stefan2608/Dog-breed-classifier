# import libraries
from keras.preprocessing import image    
from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50

from tqdm import tqdm
#from extract_bottleneck_features import * 

import pickle
import keras

import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline 

# import dog names
infile = open(dog_names,'rb')
dog_names = pickle.load(infile)

# Resnet for dog prediction
def ResNet50_predict_labels(img_path):
    ResNet50_model = ResNet50(weights='imagenet')
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))



### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    '''
    INPUT:
        'img_path' : path to a image file
    OUTPUT:
        return: "True" if a dog is detected in the image stored at img_path
    '''
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def face_detector(img_path):
    '''
    INPUT:
        'img_path' : path to a image file
    OUTPUT:
        return: "True" if a humna face in the image stored at img_path
    '''
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


# image preparation
def path_to_tensor(img_path):
        '''
    INPUT:
        'img_path' : path to a image file
    OUTPUT:
        return: transformed numpy array
    '''
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

#def paths_to_tensor(img_paths):
#    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
#    return np.vstack(list_of_tensors)

#extrect bottleneck features
def extract_Resnet50(tensor):
	from keras.applications.resnet50 import ResNet50, preprocess_input
	return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


#loading bottleneck
def Resnet50_predict_breed(img_path):
    '''
    INPUT:
        'img_path' : path to a image file
    OUTPUT:
        return: prediction of dog breed 
    '''
    # loading trained model
    Resnet50_model = keras.models.load_model('data/Resnet50.pkl')
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

# dog prediction 
def dog_breed_prediction(img_path):
    '''
    INPUT:
        'img_path' : path to a image file
    OUTPUT:
        return: Image that its provided and message based on if its a human, dog or something else.
    '''
# show image
    img = Image.open(img_path)
    plt.imshow(img)
    plt.show()
    
# human or dog detector
    human = face_detector(img_path)
    dog = dog_detector(img_path)

# if clause for dog breed detection
    if dog:
        breed = Resnet50_predict_breed(img_path)
        print ("This is a dog and it's breed is {}.".format(breed))
    elif human:
        breed = Resnet50_predict_breed(img_path)
        print("This is not a dog but it's look like a {}.".format(breed))
    else:
        print("This looks niether human or dog, must be something else.")   
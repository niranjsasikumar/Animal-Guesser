import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO

model = torch.load("./model.pth")    
model.eval()

# Processes the image given by image_path so that it can be used with the model
def process_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
    width, height = img.size
    
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    
    img = np.array(img)
    img = img.transpose((2, 0, 1))
    img = img/255
    
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    
    img = img[np.newaxis,:]
    
    image = torch.from_numpy(img)
    image = image.float()
    
    return image
    

animals = {0: 'bear', 1: 'cat', 2: 'cheetah', 3: 'chicken', 4: 'cow',
           5: 'crocodile', 6: 'deer', 7: 'dog', 8: 'dolphin', 9: 'donkey',
           10: 'duck', 11: 'elephant', 12: 'fish', 13: 'fox', 14: 'frog',
           15: 'giraffe', 16: 'goat', 17: 'hamster', 18: 'hippopotamus',
           19: 'horse', 20: 'kangaroo', 21: 'lion', 22: 'lizard', 23: 'monkey',
           24: 'mouse', 25: 'octopus', 26: 'ostrich', 27: 'panda',
           28: 'penguin', 29: 'pig', 30: 'polar bear', 31: 'rabbit',
           32: 'rhinoceros', 33: 'shark', 34: 'sheep', 35: 'snake',
           36: 'squirrel', 37: 'tiger', 38: 'turtle', 39: 'zebra'}


# Returns the predicted animal given an image and the type of image (local file or from URL)
def predict(path, image_type):
    if image_type == "1":
        try:
            response = requests.get(path)
        except:
            return "Error: Invalid URL"
        try:
            image = process_image(BytesIO(response.content))
        except:
            return "Error: The provided URL does not contain an image"
    else:
        try:
            image = process_image(path)
        except:
            return "Error: The provided path does not contain an image"
    
    output = model.forward(image)
    output = torch.exp(output)
    probs, classes = output.topk(1, dim=1)
    prob = probs.item()
    animal = classes.item()
    return ("This animal is a " + animals[animal] + ', certainty: ' + str(round(prob*100, 1)) + '%')


print("Get image from:\n[1] URL\n[2] Local File")

option = input("\nSelect an option: ")

while option != "1" and option != "2":
    option = input("Invalid input.\n\nSelect an option: ")
    
if option == "1":
    prompt = "\nImage URL: "
else:
    prompt = "\nImage path: "
    
path = input(prompt)

# Continues to ask for the path of an image until there are no errors in processing the image
while "Error" in predict(path, option):
    print(predict(path, option))
    path = input(prompt)

print(predict(path, option))
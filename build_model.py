import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

###################
##  Build Model  ##
###################

model = models.densenet161(pretrained = True)

for param in model.parameters():
    param.requires_grad = False

classifier_input = model.classifier.in_features
num_labels = 40

classifier = nn.Sequential(nn.Linear(classifier_input, 1024), nn.ReLU(),
                           nn.Linear(1024, 512), nn.ReLU(),
                           nn.Linear(512, num_labels), nn.LogSoftmax(dim=1))

model.classifier = classifier

device = torch.device("cpu")
model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters())

############################
##  Obtain Training Data  ##
############################

transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

training_set = datasets.ImageFolder("./training_set", transform = transformations)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=10, shuffle=True)

###################
##  Train Model  ##
###################

epochs = 20
progress = 0
for epoch in range(epochs):
    model.train()
    
    for inputs, labels in training_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
                    
    progress += 5
    print('Training: ' + str(progress) + '% complete.')
    
torch.save(model, "./model.pth")
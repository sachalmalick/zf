from torch.utils.data import random_split
from preprocess import ZFinchTorchset
import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn
import torch
import constants as const

class ZFinchCNN (nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Second Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Second Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # Final output
        return x
    
    def train(self, train_dl):
        # Loss Function, Optimizer and Scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(),lr=0.001)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=const.TRAINING_EPOCHS,
                                                    anneal_strategy='linear')

        # Repeat for each epoch
        for epoch in range(const.TRAINING_EPOCHS):
            running_loss = 0.0
            correct_prediction = 0
            total_prediction = 0

            print("Beginning iterative process")
            # Repeat for each batch in the training set
            for i, data in enumerate(train_dl):
                # Get the input features and target labels, and put them on the GPU
                inputs, labels = data[0], data[1]

                # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                # Zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Keep stats for Loss and Accuracy
                running_loss += loss.item()

                # Get the predicted class with the highest score
                _, prediction = torch.max(outputs,1)
                # Count of predictions that matched the target label
                correct_prediction += (prediction == labels).sum().item()
                total_prediction += prediction.shape[0]

                if i % 10 == 0:    # print every 10 mini-batches
                   print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

            # Print stats at the end of the epoch
            num_batches = len(train_dl)
            avg_loss = running_loss / num_batches
            acc = correct_prediction/total_prediction
            print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

            print('Finished Training')

def setup_and_train_model(data_processor):
    model = ZFinchCNN()
    dataset = ZFinchTorchset(data_processor)

    num_items = len(dataset)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(dataset, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

    model.train(train_dl)
    return model, train_dl, val_dl


def evaluate_model(model, val_dl):
    correctly_predicted = 0
    total_predicted = 0

    with torch.no_grad():
        for data in val_dl:
            inputs, labels = data[0], data[1]
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correctly_predicted += (prediction == labels).sum().item()
            total_predicted += prediction.shape[0]
        
    acc = correctly_predicted/total_predicted
    return acc, total_predicted

def train_and_evaluate(data_processor):
    print("Beginning training!")
    model, train_ds, eval_ds = setup_and_train_model(data_processor)
    print("Evaluating model")
    acc, total_predicted = evaluate_model(model, eval_ds)
    print(f'Accuracy: {acc:.2f}, Total items: {total_predicted}')
    torch.save(model.state_dict(), "model_state.txt")


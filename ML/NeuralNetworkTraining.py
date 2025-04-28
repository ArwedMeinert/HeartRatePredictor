import torch
import torch.nn as nn
import os
import json
import torch
import TrainingData
import GenerateData
from sklearn.model_selection import train_test_split

import random
import torch
import torch.nn as nn

class HRPredictor(nn.Module):
    def __init__(self, input_size=68, hidden_sizes=[512, 248, 124, 64]):
        """
        input_size: number of input features
        hidden_sizes: list with number of neurons in each hidden layer
        """
        super(HRPredictor, self).__init__()
        
        layers = []
        in_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())  # activation after each layer
            in_size = hidden_size
        
        # Final output layer
        layers.append(nn.Linear(in_size, 1))
        
        # Combine into a single sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def generate_parameters():
    layers=int(random.triangular(2,6,3))
    nodes=[]
    for layer in range(layers):
        nodes.append(2**(layer+6))
    nodes.reverse()
    print(nodes)
    lr=random.triangular(0.001,0.02,0.01)
    step_size=int(random.triangular(4,40,20))
    gamma=random.triangular(0.05,1,0.8)
    return nodes,lr,step_size,gamma
    
if __name__ == '__main__':
    X_data = []
    y_data = []
    X_data_verification = []
    y_data_verification = []

    # Load training and validation datasets
    training, _, _ = TrainingData.create_datasets("TrainingFiles", 100, 0)

    # Load training data
    for filename in training:
        with open(filename, "r") as file:
            data = json.load(file)
            length = data["watts"]["original_size"]
            dataGen = GenerateData.generateData(length, data)

            for t in range(length):
                dataGen.generate_data(t)

                if t > 30:  # Only after 30s we have enough history
                    power_seq = list(dataGen.last30s)
                    cadence_seq = list(dataGen.last100_cadence)[-30:]
                    elapsed_time = dataGen.elapsed_time
                    np=dataGen.np
                    gradient10=dataGen.gradient10
                    gradient30=dataGen.gradient30
                    averages = [
                        dataGen.avg_05,
                        dataGen.avg_1,
                        dataGen.avg_5,
                        dataGen.avg_10
                    ]
                    avg_hr=dataGen.past_hr
                    input_features = power_seq + cadence_seq + averages + [elapsed_time]+[np]+[gradient10,gradient30]

                    # Save input and correct heart rate
                    X_data.append(input_features)
                    y_data.append(data["heartrate"]["data"][t])
                    
    # Convert to tensors
    print(len(X_data))
    X = torch.tensor(X_data, dtype=torch.float32)
    y = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)  # add dimension

    

    # Initialize model, optimizer, and loss function
    best_loss_global = float('inf')
    iteration=1
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
    nodes,lr,step_size,gamma=[256,128,64],0.01648939775763011,26,0.896597855185665#generate_parameters()
    model = HRPredictor(input_size=68,hidden_sizes=nodes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_fn = nn.MSELoss()  # Mean Squared Error

    epochs = 2200
        
    patience = 60  # Stop if validation loss doesn't improve for 10 epochs
    no_improvement_counter = 0
        
    best_loss=float('inf')
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Training step
        y_pred_train = model(X_train)
        train_loss = loss_fn(y_pred_train, y_train)

        train_loss.backward()
        optimizer.step()
        scheduler.step()

        # Validation step
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            y_pred_val = model(X_val)
            val_loss = loss_fn(y_pred_val, y_val)

        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            no_improvement_counter = 0
            best_mode_state=model.state_dict()
        else:
            no_improvement_counter += 1

        if no_improvement_counter >= patience or  (epoch>40 and best_loss>2000) or (epoch>100 and best_loss>1000):
            print("Early stopping: No improvement in validation loss.")
            break

        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}")

    # Save the model after training
    print(iteration)
    iteration+=1
    best_loss_global = best_loss
    torch.save(best_mode_state, "modelnew.pth")
            
    parameters = {
                "layers": nodes,
                "learning_rate": lr,
                "step_size": step_size,
                "gamma": gamma,
                "validation_loss": best_loss.item()
            }
            
    with open("model_parametersnew.json", "w") as f:
        json.dump(parameters, f, indent=4)
            
    print(f"A new model has been saved with layers: {nodes}, lr: {lr:.6f}, step_size: {step_size}, gamma: {gamma:.6f}. Validation loss: {best_loss:.6f}")

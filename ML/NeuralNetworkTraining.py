import torch
import torch.nn as nn
import os
import json
import torch
import TrainingData
import GenerateData

class HRPredictor(nn.Module):
    def __init__(self):
        super(HRPredictor, self).__init__()
        self.fc1 = nn.Linear(68, 256)  # Increased number of units
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 1)  # output: heart rate

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x
    
    
if __name__ == '__main__':
    X_data = []
    y_data = []
    X_data_verification = []
    y_data_verification = []

    # Load training and validation datasets
    training, verification, _ = TrainingData.create_datasets("TrainingFiles", 70, 30)

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

    # Load validation data
    for filename in verification:
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
                    input_features = power_seq + cadence_seq + averages + [elapsed_time]+[np]+[gradient10,gradient30]

                    # Save input and correct heart rate
                    X_data_verification.append(input_features)
                    y_data_verification.append(data["heartrate"]["data"][t])
                    
    # Convert to tensors
    print(len(X_data))
    X_train = torch.tensor(X_data, dtype=torch.float32)
    y_train = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)  # add dimension

    X_val = torch.tensor(X_data_verification, dtype=torch.float32)
    y_val = torch.tensor(y_data_verification, dtype=torch.float32).unsqueeze(1)

    # Initialize model, optimizer, and loss function
    model = HRPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.05)
    loss_fn = nn.MSELoss()  # Mean Squared Error

    epochs = 200
    best_loss = float('inf')
    patience = 20  # Stop if validation loss doesn't improve for 10 epochs
    no_improvement_counter = 0

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
        else:
            no_improvement_counter += 1

        if no_improvement_counter >= patience:
            print("Early stopping: No improvement in validation loss.")
            break

        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}")

    # Save the model after training
    torch.save(model.state_dict(), "model.pth")

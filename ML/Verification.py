import TrainingData
import GenerateData
import json
import matplotlib.pyplot as plt
import torch
from collections import deque
import NeuralNetworkTraining

# Assuming your trained model is saved as 'model.pth'
class SimulateActivity:
    def __init__(self, model, activity_data, window_size=30):
        self.model = model
        self.activity_data = activity_data  # This is your input data (power, cadence, heart rate)
        self.window_size = window_size  # How much history you want to keep for prediction
        self.last30s_power = deque([0]*window_size)
        self.last30s_cadence = deque([0]*window_size)
        self.averages = {
            'avg_05': 0,
            'avg_1': 0,
            'avg_5': 0,
            'avg_10': 0
        }
        self.elapsed_time=0
        self.predicted_hr = []
        self.real_hr = []
        self.gradient10=0
        self.gradient30=0

    def simulate(self):
        for t in range(len(self.activity_data["watts"]["data"])):
            # Extract current timestep data
            power = self.activity_data["watts"]["data"][t]
            cadence = self.activity_data["cadence"]["data"][t]
            heart_rate = self.activity_data["heartrate"]["data"][t]
            self.elapsed_time=self.activity_data["time"]["data"][t]
            # Update windows for power, cadence
            self.last30s_power.append(power)
            self.last30s_cadence.append(cadence)
            
            

            if len(self.last30s_power) > self.window_size:
                self.last30s_power.popleft()
                self.last30s_cadence.popleft()

            # Update averages (e.g., avg_10, avg_5, etc.)
            self.averages['avg_05'] = sum(list(self.last30s_power)[-30:]) / 30
            self.averages['avg_1'] = sum(list(self.last30s_power)[-60:]) / 60
            self.averages['avg_5'] = sum(list(self.last30s_power)[-5*60:]) / (5*60)
            self.averages['avg_10'] = sum(self.last30s_power) / (10*60)

            if self.elapsed_time == 0:
                # Initialize NP for the first timestep
                self.np = self.averages['avg_05']**0.25  # Or you can set it based on a default value
            else:
                # Incrementally update NP using previous NP
                self.np = ((self.np**4 * self.elapsed_time) + self.averages['avg_05']**4) / (self.elapsed_time + 1)
                self.np = self.np**0.25
            
            if self.elapsed_time<10:
                self.gradient10=power-self.last10[0]
                self.gradient30=power-self.last10[0]
            elif self.elapsed_time<30:
                self.gradient10=power-self.last10[-10]
                self.gradient30=power-self.last10[0]
            else:
                self.gradient10=power-self.last10[-10]
                self.gradient30=power-self.last10[-30]
            
            # Prepare input for model (current power, cadence, and averages)
            input_features = list(self.last30s_power) + list(self.last30s_cadence)[-30:] + list(self.averages.values())+[self.elapsed_time]+[self.np]

            # Convert to tensor and predict
            x_input = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                predicted_hr = self.model(x_input).item()

            # Store real and predicted heart rate
            self.real_hr.append(heart_rate)
            self.predicted_hr.append(predicted_hr)

        self.plot_results()

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.real_hr, label="Real Heart Rate", color='blue')
        plt.plot(self.predicted_hr, label="Predicted Heart Rate", color='red', linestyle='--')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Heart Rate (bpm)")
        plt.title("Real vs Predicted Heart Rate")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    # Load your trained model
    model = NeuralNetworkTraining.HRPredictor()  # Replace HRPredictor with your actual model class
    model.load_state_dict(torch.load('model.pth'))
    model.eval()  # Set model to evaluation mode
    
    # Create dataset
    training, verification, testing = TrainingData.create_datasets("TrainingFiles", 70, 20, 10)
    
    # Select an activity from the dataset (e.g., first training file)
    with open(verification[1], "r") as file:
        data = json.load(file)
        length_activity = data["watts"]["original_size"]
        
        # Initialize data generator (this will be used for generating features)
        dataGenerator = GenerateData.generateData(length_activity, data)

        # Simulate the activity and plot results
        simulator = SimulateActivity(model, data)
        simulator.simulate()  # This will run the simulation and plot the results

import json
import TrainingData
from collections import deque

class generateData:
    def __init__(self, length: int, data: json):
        self.data=data
        self.length = length
        self.avg = 0
        self.np = 0
        self.avg_10 = 0
        self.avg_5 = 0
        self.avg_1 = 0
        self.avg_05 = 0
        self.last10 = deque([0] * (10 * 60))  # Last 10 minutes (assuming 1 value/sec)
        self.last30s = deque([0] * 30)        # For NP: Last 30 seconds of power
        self.last100_cadence = deque([0] * 100)
        self.elapsed_time = 0
        self.np_samples = []                  # Store 30s averages for NP calculation
        self.past_hr=0
        self.gradient10=0
        self.gradient30=0

    def generate_data(self, timestep: int):
        current_power = self.data["watts"]["data"][timestep]
        current_cadence = self.data["cadence"]["data"][timestep]
        current_time = self.data["time"]["data"][timestep]

        # Update overall average correctly
        if timestep == 0:
            self.avg = current_power
        else:
            self.avg = ((self.avg * timestep) + current_power) / (timestep + 1)
            
        # Update last 10 minutes for time-based averages
        self.last10.append(current_power)
        self.last10.popleft()

        # Update last 30 seconds for NP
        self.last30s.append(current_power)
        self.last30s.popleft()

        avg30s = sum(self.last30s) / len(self.last30s)
        self.np_samples.append(avg30s)
        
        if timestep == 0:
            # Initialize NP for the first timestep
            self.np = avg30s**0.25  # Or you can set it based on a default value
            self.gradient10=current_power
            self.gradient30=current_power
        else:
            # Incrementally update NP using previous NP
            self.np = ((self.np**4 * timestep) + avg30s**4) / (timestep + 1)
            self.np = self.np**0.25

        if timestep<10:
            self.gradient10=current_power-self.last10[0]
            self.gradient30=current_power-self.last10[0]
        elif timestep<30:
            self.gradient10=current_power-self.last10[-10]
            self.gradient30=current_power-self.last10[0]
        else:
            self.gradient10=current_power-self.last10[-10]
            self.gradient30=current_power-self.last10[-30]
        # Update 30s, 1min, 5min, and 10min averages
        self.avg_05 = sum(list(self.last10)[-30:]) / 30    # 30 seconds
        self.avg_1 = sum(list(self.last10)[-60:]) / 60      # 1 minute
        self.avg_5 = sum(list(self.last10)[-5*60:]) / (5*60)  # 5 minutes
        self.avg_10 = sum(self.last10) / (10*60)            # 10 minutes

        # Update cadence
        self.last100_cadence.append(current_cadence)

        # Update elapsed time
        self.elapsed_time = current_time
    
    
if __name__ == '__main__':
    
    training, verification, testing = TrainingData.create_datasets("TrainingFiles", 70, 20, 10)
    
    with open(training[1],"r") as file:
        data=json.load(file)
        length_activity=data["watts"]["original_size"]
        dataGenerator=generateData(length_activity,data)
        for i in range(length_activity):
            dataGenerator.generate_data(i)
            print(dataGenerator.np)
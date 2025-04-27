import json
import os
import random

def create_datasets(directory: str, training_amount: int, verification_amount: int, testing_amount: int = None) -> list:
    if sum([training_amount, verification_amount, testing_amount or 0]) > 100:
        raise Exception("The overall amount should not be higher than 100%")
    
    training = []
    verification = []
    testing = []

    for file in os.scandir(directory):
        if file.is_file():
            with open(file.path, "r") as f:
                data = json.load(f)
                
            if "heartrate" in data and "watts" in data and "cadence" in data and "time" in data and data["watts"]["original_size"]>600 and sum(data["heartrate"]["data"][1:30])/30<140:
                r = random.random()
                if r < training_amount / 100:
                    training.append(file.path)
                elif r < (training_amount + verification_amount) / 100:
                    verification.append(file.path)
                else:
                    testing.append(file.path)

    return training, verification, testing

if __name__ == '__main__':
    training, verification, testing = create_datasets("TrainingFiles", 70, 20, 10)

    # Open the second training file
    with open(training[1], "r") as file:
        data = json.load(file)
        print(data["watts"]["data"][1])
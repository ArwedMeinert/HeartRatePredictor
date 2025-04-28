# Heart Rate Predictor
The goal of this project is to predict heart rate during cycling based on various averages of power and cadence. A neural network was trained for this purpose.

## Data Aquisition
Data for power, cadence, and heart rate is available. To obtain it, the Strava API was used to access recent activities, filter them by activity type (indoor cycling), and download the relevant data as a JSON file.
```python
def get_activities(token:str, client_id:str, client_secret:str, refresh_token:str,amount:int,page:int):
    # Optional query parameters
    params = {
        "page": page,
        "per_page": amount
    }

    # Authorization headers
    headers = {
        "Authorization": f"Bearer {token}"
    }

    try:
        activities = requests.get(
            "https://www.strava.com/api/v3/athlete/activities",
            headers=headers,
            params=params
        )
        activities.raise_for_status()  # Will throw an error if unauthorized (401)
    except requests.exceptions.HTTPError as e:
        if activities.status_code == 401:
            print("Access token expired, refreshing...")
            # Get new token
            token = get_refresh_token(client_id, client_secret, refresh_token)
            # Update headers with new token
            headers["Authorization"] = f"Bearer {token}"
            # Retry the request
            activities = requests.get(
                "https://www.strava.com/api/v3/athlete/activities",
                headers=headers,
                params=params
            )
            activities.raise_for_status()
        else:
            # Re-raise if it's not a 401 error
            raise e

    # If successful
    return activities.json(),token
```
The power, cadence, elapsed time, and heart rate are saved into a JSON file named after the activity. This allows for building a large dataset.
### Considerations of the Dataset
Since heart rate is highly individual and not solely dependent on power output, the dataset should not be too large. It should include only the last few weeks of activities to avoid excessive variation.

## Data Generation
Since heart rate mainly depends on power output, the last 30 seconds of data are used as inputs. Additionally, moving averages over 30 seconds, 1 minute, 5 minutes, and 10 minutes are calculated, along with the overall average and normalized power. Cadence over the last 30 seconds is also included.

## Neural Network
To find the best neural network parameters, several permutations with low epochs were tested. The best-performing configuration was then trained longer.
```python
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
```
The final chosen parameters are:
```python
nodes,lr,step_size,gamma=[256,128,64],0.01648939775763011,26,0.896597855185665
```
The neural network has 68 input nodes, three hidden layers (256, 128, and 64 nodes respectively), and a learning rate that decreases after every 26 epochs.
Validation was done using a separate validation dataset. In final training, the loss converged around 100, reflecting the dataset's unstructured nature and heart rate variability. Factors such as improved fitness (lower heart rate) or illness (higher heart rate) make training challenging. 
```python
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
```
## Results
The final model performs well during steady rides but struggles with sudden variations. Importantly, the input parameters can be gathered live while riding, enabling heart rate prediction without a heart rate monitor.

<img src="https://github.com/user-attachments/assets/9f458989-1138-450f-a3f5-ee7649567131" alt="Steady ride prediction" style="width:60%" />

The diagram above shows a relatively steady ride. Although the predicted heart rate is slightly lower, it still captures the general trend.

<img src="https://github.com/user-attachments/assets/8009f4d6-b1b3-4f34-947b-e05538de4a38" alt="High-intensity prediction" style="width:60%" />

This training session had more variation. While the model predicts the initial heart rate well, it struggles during high-intensity sections, likely due to a lack of high-intensity data.

<img src="https://github.com/user-attachments/assets/1c9908e0-f5f3-42c2-9499-6a5a1c2dc558" alt="Offset prediction" style="width:60%" />

Here, the model captures the heart rate trend well but shows a constant offset. This likely results from the model being trained on activities with generally lower heart rates.
## Futrure Work
The next step is real-time use during cycling. Power and cadence data can be collected via Bluetooth to predict heart rate without a sensor. Although not perfectly accurate, the prediction is still a valuable alternative to having no heart rate data at all. 

import requests
import os
import json

CLIENT_ID = "53436"
CLIENT_SECRET = "7cb942d9f4ee69e11b3268f08219d5893e84425f"
REDIRECT_URI = "http://localhost/exchange_token"  # You can use anything, but must match app settings!

# Scope needed: activity:read_all to access all activities
SCOPE = "activity:read_all"


def get_refresh_token(clientID:str, clientSecret:str, refreshToken:str):
    response = requests.post(
        "https://www.strava.com/api/v3/oauth/token",
        data={
            "client_id": clientID,
            "client_secret": clientSecret,
            "grant_type": "refresh_token",
            "refresh_token": refreshToken,
        }
    )
    response.raise_for_status()
    return response.json()["access_token"]  # Return only the access token

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

def filter_by_virtual(activities:json,previous_activities=None)->list:
    if previous_activities==None:
        previous_activities=[]
    for activity in activities:
        if activity["type"]=="VirtualRide":
            previous_activities.append(activity["id"])
    return previous_activities


def get_activity(activity_id: int, location: str, token: str) -> None:
    """Download a Strava activity and save it as a JSON file if it doesn't exist yet."""
    
    # Ensure directory exists
    os.makedirs(location, exist_ok=True)
    
    # Build filename
    filename = os.path.join(location, f"{activity_id}.json")
    
    # Check if the file already exists
    if os.path.exists(filename):
        print(f"Activity {activity_id} already exists at {filename}.")
        return
    
    # If not, download it
    url = f"https://www.strava.com/api/v3/activities/{activity_id}?include_all_efforts="
    headers = {
        "Authorization": f"Bearer {token}"
    }

    response = requests.get(url, headers=headers)
    
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"Failed to fetch activity {activity_id}: {e}")
        return

    # Save JSON to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(response.json(), f, indent=4)

    print(f"✅ Activity {activity_id} downloaded and saved to {filename}.")

def download_streams(activity_id: int, location: str, token: str) -> None:
    """Download a Strava activity streams (HR, cadence, power) as a JSON file if it doesn't already exist."""

    # Ensure the directory exists
    os.makedirs(location, exist_ok=True)

    # Build filename
    filename = os.path.join(location, f"{activity_id}.json")

    # Check if the file already exists
    if os.path.exists(filename):
        print(f"Streams for activity {activity_id} already exist at {filename}.")
        return

    # Build request
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    params = {
        "keys": "time,heartrate,cadence,watts",
        "key_by_type": "true"
    }
    headers = {
        "Authorization": f"Bearer {token}"
    }

    response = requests.get(url, headers=headers, params=params)

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"Failed to fetch streams for activity {activity_id}: {e}")
        return

    # Save the JSON file
    with open(filename, "w") as file:
        json.dump(response.json(), file, indent=2)

    print(f"✅ Streams for activity {activity_id} downloaded and saved to {filename}.")
    
    
if __name__ == '__main__':
    client_id = "CLIENTID"
    token = 'TOKEN'
    client_secret = "CLIENT_SECRET"
    refresh_token = "REFRESH_TOKEN"
    for i in range(3):
        activities,token=get_activities(token, client_id, client_secret, refresh_token,amount=50,page=i+1)
        indoor_rides=filter_by_virtual(activities)
        
        
    print(indoor_rides)
    for ride in indoor_rides:
        download_streams(ride,"TrainingFiles",token)
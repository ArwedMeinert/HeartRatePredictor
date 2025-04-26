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
    print(activities.json())  # Assuming you want the JSON response
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

def download_gpx(activity_id: int, location: str, token: str) -> None:
    """Download a Strava activity as a GPX file if it doesn't already exist."""

    # Ensure the directory exists
    os.makedirs(location, exist_ok=True)

    # Build filename
    filename = os.path.join(location, f"{activity_id}.gpx")

    # Check if the file already exists
    if os.path.exists(filename):
        print(f"GPX for activity {activity_id} already exists at {filename}.")
        return

    # Build request
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/export_gpx"
    headers = {
        "Authorization": f"Bearer {token}"
    }

    response = requests.get(url, headers=headers)

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"Failed to fetch GPX for activity {activity_id}: {e}")
        return

    # Save the GPX file
    with open(filename, "wb") as f:
        f.write(response.content)

    print(f"✅ GPX for activity {activity_id} downloaded and saved to {filename}.")
    
if __name__ == '__main__':
    client_id = "53436"
    code = "90f240441b713c31cbd60fd58842a0375a560e0e"  # Replace with the code you copied!
    token = 'fef948654a5cd2f4de6082cce60013ccc2a7f38a'

    client_id = "53436"
    client_secret = "7cb942d9f4ee69e11b3268f08219d5893e84425f"
    refresh_token = "3bd80c0a333776af6fc093768cac257533ac62cf"
    activities,token=get_activities(token, client_id, client_secret, refresh_token,amount=100,page=1)
    indoor_rides=filter_by_virtual(activities)
    print(indoor_rides)
    download_gpx(indoor_rides[0],"TrainingFiles",token)
    
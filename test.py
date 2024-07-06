import requests
import json

url = 'http://127.0.0.1:5000/predict'
data = [
    {
        "gf_rolling": 1.5, 
        "ga_rolling": 0.5, 
        "sh_rolling": 8, 
        "sot_rolling": 3, 
        "pk_rolling": 0.2, 
        "pkatt_rolling": 0.5, 
        "saves_rolling": 3, 
        "cs_rolling": 1,
        "venue_code": 1, 
        "opp_code": 2, 
        "hour": 21, 
        "day_code": 5
    }
]

headers = {'Content-Type': 'application/json'}
response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())

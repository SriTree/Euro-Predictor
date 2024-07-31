# UEFA 2024 Predictor (Backend)

Hi! My name is Sri, and this is my project to predict the UEFA 2024 Euros. As someone who tries to catch every soccer match, this project seemed like the perfect combination of sports analytics and machine learning.

This repository is split into three components: data, notebooks, and the `server.py` file for my backend.

- **Notebooks**: Contains all my test notebooks for the final model that utilizes XGBoost. Additionally, I have included `scrape.ipynb`, which contains the code for data extraction from [FBref](https://fbref.com/en/). Everything is well-documented so that someone could create their own model.
- **Data**: Contains the data compiled from the scrape notebook in CSV format.
- **Server.py**: Sets up my basic Flask app and route to be accessed by the frontend.

## Deployment

To deploy this project, run:

```bash
python server.py
```

Additionally I chose to deploy this backend to [render](https://render.com/), but you can deploy it anywhere you would like to.
## Acknowledgements

 - [How to predict NFL Winners with Python](https://www.activestate.com/blog/how-to-predict-nfl-winners-with-python/)
 - [Predict Football Match Winners With Machine Learning And Python](How to predict NFL Winners with Python)

## Tech Stack

**Server:** Flask

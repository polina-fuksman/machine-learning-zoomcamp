# Delivery Time Prediction

This project focuses on predicting the time required for a courier to deliver an order. By analyzing various features related to the order, courier, and environmental conditions, the model provides an estimated delivery time in minutes.

The primary goal was to explore the factors that influence delivery time and to build an effective regression model using XGBoost.

The final model is an **XGBoost Regressor**, which achieved a **Root Mean Squared Error (RMSE) of 3,7 minutes** on the test set.

## Dataset

The model was trained on the [Food-Delivery dataset](https://huggingface.co/datasets/aneesarom/Food-Delivery) from Hugging Face. This dataset contains anonymized, real-world data from a food delivery service.

## Features

The following features were used to train the final model:
*   `delivery_person_age`
*   `delivery_person_ratings`
*   `weather_conditions`
*   `road_traffic_density`
*   `vehicle_condition`
*   `type_of_order`
*   `type_of_vehicle`
*   `multiple_deliveries`
*   `festival`
*   `city`
*   `distance_km`
*   `order_hour`
*   `order_day_of_week`
*   `preparation_time_mins`

## How to Run the Project

You can run this project in two ways: using Docker or setting up the environment manually.

### Option 1: Running with Docker (Recommended)

The project includes a `Dockerfile` to easily build and run the prediction service in a containerized environment.

**Prerequisites:**
*   [Docker](https://www.docker.com/products/docker-desktop/) installed and running.

**Steps:**

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/polina-fuksman/machine-learning-zoomcamp.git
    cd machine-learning-zoomcamp/07-midterm-project
    ```

2.  **Build the Docker Image**
    From the `07-midterm-project` directory, build the Docker image:
    ```bash
    docker build -t delivery-time-prediction .
    ```
    It will generate the `model_eta=0.1_max_depth=10_min_child_weight=1.bin` artifact, the trained model.

3.  **Run the Docker Container**
    Run the container, mapping port `9696` on your local machine to the container's port `9696`:
    ```bash
    docker run -it -p 9696:9696 delivery-time-prediction
    ```
    The prediction service is now running and accessible at `http://localhost:9696`.


### Option 2: Manual Setup

Follow these instructions to set up the environment, train the model, and run the prediction service from scratch.

**Prerequisites:**
*   Python 3.9
*   [pipenv](https://pipenv.pypa.io/en/latest/installation/)

**Steps:**

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/polina-fuksman/machine-learning-zoomcamp.git
    cd machine-learning-zoomcamp/07-midterm-project
    ```

2.  **Set Up the Environment**
    Create the virtual environment and install the required packages from the `Pipfile`:
    ```bash
    pipenv install
    ```

3.  **Run the Prediction Service**
    Start the Flask service using Gunicorn:
    ```bash
    pipenv run gunicorn --bind 0.0.0.0:9696 predict:app
    ```
    The service will now be running and listening for requests on port 9696.

### Getting Predictions

Once the service is running (either via Docker or manually), you can test the prediction endpoint using the provided `test.py` script.

1.  Make sure the prediction service is running in one terminal.
2.  Open a **new terminal** window and navigate to the project directory (`07-midterm-project`).
3.  If you set up the project manually, ensure your `pipenv` shell is active.
4.  Run the test script:
    ```bash
    python test.py
    ```
This script will send a sample order to the prediction service and print the JSON response.

**Example Response:**
```json
  {'delivery_time_prediction_in_min': 21.95}
    Will be at the address in 23.0 minutes
```


## Image of how you interact with the deployed service

Can be found here [test.ipynb](https://github.com/polina-fuksman/machine-learning-zoomcamp/blob/main/07-midterm-project/test.ipynb)

Process of training and finding the best model for the task can be found here [midterm-project.ipynb](https://github.com/polina-fuksman/machine-learning-zoomcamp/blob/main/07-midterm-project/midterm-project.ipynb)


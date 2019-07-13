# Disaster Response Pipeline Project
This project has the intent of representing a full ETL piepline from raw data through to a finished interactive webpage app. The app itself analyses text inputs from users in order to classify potential factors for diagnosing responses in areas hit by natural disasters. It does this by utilising a RandomForect multi-output classifier model trained through the ETL pipeline process.

### Required installations:
- sys
- pandas
- numpy
- sqlalchemy
- nltk
- plotly
- sklearn
- pickle

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# Disaster Response Pipeline Project
This project has the intent of representing a full ETL piepline from raw data through to a finished interactive webpage app. The app itself analyses text inputs from users in order to classify potential factors for diagnosing responses in areas hit by natural disasters. It does this by utilising a RandomForect multi-output classifier model trained through the ETL pipeline process.

### Required installations:
Testing has been completed on Python 3.6.3. Some testing has been completed on Python 3.7 but I cannot attest to it's reliability at this point. Brackets next to libraries below for fully tested versions. 
- sys
- pandas (0.23.3)
- numpy (1.12.1)
- sqlalchemy (1.2.18)   
- nltk (3.2.5)
- plotly (2.0.15)
- sklearn (0.19.1, also tested 0.23)
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

### Modules
#### process_data.py
This module takes two input csv files, disaster_messages (containing raw text of messages) and disaster_categories (containing a list of classifications of these messages, referenced by ID, indicating which "response" categories apply to the messages), and outputs a single database for later use, with data cleaned and transformed to remove duplicates, effectively onehotencode categorical variables and merge the datasets.

#### train_classifier.py
This module uses the database created using process_data in order to create a a RandomForect multi-output classifier model for later use in classifying text inputs into the same categories used in the disaster_categories.csv file.

#### run.py
Runs a web app allowing user input of text to show classification per the disaster_categories framework. The web app also shows some basic statistics on first logging in.

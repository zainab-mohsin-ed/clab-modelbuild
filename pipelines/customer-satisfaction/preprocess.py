import argparse
import logging
import pathlib

import boto3
import numpy as np
import pandas as pd
import pickle
import os

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.info("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    print(input_data)
    bucket = input_data.split("/")[0]
    key = "/".join(input_data.split("/")[1:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/raw-data.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.info("Reading downloaded data.")
    df = pd.read_csv(fn)
    df.rename(columns={'Customer Type':'Customer_Type','Type of Travel':'Type_of_Travel','Flight Distance':'Flight_Distance','Inflight wifi service':'Inflight_wifi_service','Departure/Arrival time convenient':'Departure/Arrival_time_convenient','Ease of Online booking':'Ease_of_Online_booking','Gate location':'Gate_location','Food and drink':'Food_and_drink','Online boarding':'Online_boarding','Seat comfort':'Seat_comfort','Inflight entertainment':'Inflight_entertainment','Leg room service':'Leg_room_service','Baggage handling':'Baggage_handling','Checkin service':'Checkin_service','Inflight service':'Inflight_service','Departure Delay in Minutes':'Departure_Delay_in_Minutes', 'On-board service':'On_board_service'},inplace=True)

    # Initialize LabelEncoder
    label_encoders = {}
    label_cols = ["Gender", "satisfaction", "Customer_Type", "Type_of_Travel"]
    
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  
    
    # One-Hot Encoding for Flight Class
    df = pd.get_dummies(df, columns=["Class"], prefix="Class", dtype=int)
    
    X = df.drop('satisfaction', axis =1)
    y = df['satisfaction']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.4)

    # Split the data
    pd.DataFrame(np.c_[y_train, X_train]).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(np.c_[y_val, X_val]).to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    pd.DataFrame(np.c_[y_test, X_test]).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
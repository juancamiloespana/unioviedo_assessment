
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import  cross_val_score, cross_validate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf



def compair_models(modelos,scoring,X,y,cv):

    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        scores=cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos,pdscores],axis=1)
    
    metric_modelos.columns=["reg_lineal","decision_tree","random_forest","gradient_boosting"]
    return metric_modelos




def generate_medical_ts(n_users=2000, n_timesteps=50, SEED=123, prop_iterval=0.8):
   
    # Generate synthetic medical time series data (Vitals and Progression Labels)
    # and uses Random Forest to create progression labels based on the features.
    # Returns a DataFrame with patient-wise time series and disease progression labels.
       
    # Feature names: [HeartRate, BloodPressure, RespRate, Temperature, WBC, Glucose]
    
    features = ["HeartRate", "BloodPressure", "RespRate", "Temperature", "WBC", "Glucose"]
    
    patient_data = []

    # First,  generate random vitals for each patient
    for patient_id in range(n_users):
        interv=np.round(n_timesteps*prop_iterval).astype(int)  # Interval for random time steps
      

        coef_mean=np.random.normal(loc=0, scale=1)
        coef_var=np.random.uniform(0, 1)
        
        # Add the generated data for this patient
        for t in range(n_timesteps):
            
            if random.random() > prop_iterval:
                vital = [np.nan] * len(features)  # Assign missing values
            else:
                vital=np.random.normal(loc=t*coef_mean, scale=1+t*coef_var, size=len(features))
            # Generate random vitals for each featur
            patient_data.append({
                "patient_id": patient_id+1,
                "timestamp": t+1,
                #"coef":coef_mean,
                **{features[i]: vital[i] for i in range(len(features))}  # Create individual columns for each feature
            })
    # Convert the data into a DataFrame for Random Forest Training
    df = pd.DataFrame(patient_data)
    return df

df=generate_medical_ts(n_users=10,n_timesteps=10,SEED=123,prop_iterval=0.8)

# df.head(60)


def df_to_ts_arrays(df):


    df = df.sort_index()
    feat_labels=['HeartRate', 'BloodPressure', 'RespRate', 'Temperature', 'WBC', 'Glucose']

    X = []  # Features

    #ids users
    ids= df['patient_id'].unique()
    timestamps = df['timestamp'].unique()

  
    for patient in ids:
        patient_data = df[df['patient_id'] == patient]
        features = patient_data[feat_labels].values
        X.append(features)
      
   
    X = np.array(X)  # Shape: (n_samples, 30, 6) where 30 is time_steps and 6 is the number of features
  
    return X, ids,timestamps,feat_labels



def transf_inputs(df, SEED=123):

    
    tf.config.experimental.enable_op_determinism() ## make the code reproducible
    
    df=df.reset_index(drop=False)
    X, ids, timestamps, feat_labels = df_to_ts_arrays(df)

    n=df.shape[0]
    


    n_timesteps = X.shape[1]  # 30 time steps
    n_features = X.shape[2]   # 6 features per timestep


    # Define the model using Sequential
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(n_timesteps,n_features)),  # 30 time steps, 6 features per time step
        Dropout(0.2), 
        LSTM(32, return_sequences=True),
        Dense(32, activation='relu'),  # Fully connected layer 
        Dense(3, activation='softmax')  # 3 classes: Stable, Progressing, Critical
    ])

    
    y_prob= model.predict(X)
    ## introduce randomenss
    perturbation = np.random.normal(0, 0.001, size=y_prob.shape)  # Gaussian noise with mean 0 and std 0.01
    y_prob_perturbed = y_prob + perturbation

    y_prob_perturbed /= y_prob_perturbed.sum(axis=-1, keepdims=True)


    y_pred = np.argmax(y_prob_perturbed, axis=-1)
   
    label_map = {0: "Stable", 1: "Progressing", 2: "Critical"}

    vectorized_map = np.vectorize(label_map.get)

    y_pred_labels = vectorized_map(y_pred)

    df['label']=y_pred_labels.reshape(-1,1)

    return df, X, y_pred_labels, ids, timestamps, feat_labels


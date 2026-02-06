import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

def generate_data(n_samples=1000):
    np.random.seed(42)
    
 
    data = {
        'screen_time': np.random.normal(8, 3, n_samples),
        'social_media_time': np.random.normal(3, 1.5, n_samples),
        'gaming_time': np.random.normal(1.5, 1, n_samples),
        'notifications': np.random.normal(150, 50, n_samples),
        'phone_pickups': np.random.normal(70, 30, n_samples),
        'sleep_hours': np.random.normal(7, 1.5, n_samples),
        'age': np.random.randint(15, 60, n_samples),
        'anxiety_level': np.random.randint(1, 11, n_samples),
        'feel_anxious': np.random.randint(0, 4, n_samples),
        'interrupt_sleep': np.random.randint(0, 4, n_samples),
        'neglect_responsibilities': np.random.randint(0, 4, n_samples),
        'failed_reduce': np.random.randint(0, 4, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    cols = ['screen_time', 'social_media_time', 'gaming_time', 'notifications', 'phone_pickups', 'sleep_hours']
    for col in cols:
        df[col] = df[col].clip(lower=0)
 
    def determine_risk(row):
        score = 0
        if row['screen_time'] > 8: score += 2
        if row['social_media_time'] > 3: score += 3
        if row['gaming_time'] > 3: score += 1
        if row['notifications'] > 200: score += 1
        if row['phone_pickups'] > 100: score += 1
        if row['sleep_hours'] < 6: score += 3
        if row['anxiety_level'] > 7: score += 1
        if row['feel_anxious'] >= 2: score += 2
        if row['interrupt_sleep'] >= 2: score += 2
        if row['neglect_responsibilities'] >= 2: score += 2
        if row['failed_reduce'] >= 2: score += 1
        
        if score < 5:
            return 0
        elif score < 10 and score >=5:
            return 1
        else:
            return 2

    df['risk_level'] = df.apply(determine_risk, axis=1)
    
    return df

def train_model():
    print("Loading real dataset...")
     
    import os
    possible_paths = [
        'mobile_addiction_data.csv',
        os.path.join(os.path.dirname(__file__), 'mobile_addiction_data.csv'),
        os.path.join('venv', 'mobile_addiction_data.csv')
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Loaded dataset from {path}")
            break
    
    if df is None:
        raise FileNotFoundError("Could not find mobile_addiction_data.csv")

    
    def determine_risk(row):
        score = 0
        if row['Screen_Time'] > 6: score += 2
        if row['Social_Media_Usage'] > 2: score += 3
        if row['Gaming_Hours'] > 1: score += 1
        if row['Notifications_Per_Day'] > 100: score += 1
        if row['Phone_Unlocks'] > 60: score += 1
        if row['Sleep_Hours'] < 7: score += 2
        if row['Anxiety_Score'] > 50: score += 1
        
        if score < 4: return 0
        elif score < 8: return 1
        else: return 2

    df['risk_level'] = df.apply(determine_risk, axis=1)
    
   
    feature_cols = ['Age', 'Screen_Time', 'Phone_Unlocks', 'Social_Media_Usage', 
                    'Gaming_Hours', 'Sleep_Hours', 'Anxiety_Score', 'Notifications_Per_Day']
    
    X = df[feature_cols]
    y = df['risk_level']
  
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print("Training Gradient Boosting model...")
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    
    from sklearn.metrics import classification_report, accuracy_score
    y_pred = model.predict(X_test)
    
  
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
     
    print("Saving model and scaler to addiction_model.pkl...")
    import os
    save_path = os.path.join(os.path.dirname(__file__), 'addiction_model.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
 

if __name__ == "__main__":
    train_model()

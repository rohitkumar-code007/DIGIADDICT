import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

 
st.set_page_config(
    page_title="Digital Addiction Risk Predictor",
    layout="wide"
)
 
@st.cache_resource
def load_model_and_scaler():
    try:
        import os
        model_path = os.path.join(os.path.dirname(__file__), 'addiction_model.pkl')
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['scaler']
    except FileNotFoundError:
        st.error("Model file not found. Please train the model first by running train_model.py")
        return None, None
    except KeyError:
        st.error("Model file corrupted or old format. Please regenerate model.")
        return None, None

 
def main():
    st.markdown("<h1 style='text-align: center;'>Digital Addiction Risk Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Assess your digital device usage patterns</h3>", unsafe_allow_html=True)
    
    model, scaler = load_model_and_scaler()
    
    if model is None:
        st.stop()
    
  
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Usage Statistics")
        screen_time = st.slider("Daily screen time (hours)", 0.0, 24.0, 6.0)
        social_media_time = st.slider("Social media time (hours/day)", 0.0, 12.0, 2.0)
        gaming_time = st.slider("Gaming time (hours/day)", 0.0, 12.0, 1.0)
        notifications = st.number_input("Daily notifications received", 0, 1000, 100)
        
    with col2:
        st.subheader("Behavioral Patterns")
        phone_pickups = st.number_input("Times you check phone daily (Unlocks)", 0, 500, 50)
        sleep_hours = st.slider("Average sleep (hours)", 0.0, 12.0, 7.0)
        age = st.number_input("Your age", 10, 100, 25)
        anxiety_level = st.slider("Anxiety Score (0-100)", 0, 100, 50) 
        
    feature_names = ['Age', 'Screen_Time', 'Phone_Unlocks', 'Social_Media_Usage', 
                     'Gaming_Hours', 'Sleep_Hours', 'Anxiety_Score', 'Notifications_Per_Day']
    
    features_df = pd.DataFrame([[
        age, screen_time, phone_pickups, social_media_time, 
        gaming_time, sleep_hours, anxiety_level, notifications
    ]], columns=feature_names)
    
 
    features_scaled = scaler.transform(features_df)
     
    if st.button("Analyze Risk", type="primary"):
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        st.markdown("---")
        st.subheader("Results")

        risk_messages = {
            0: ("**Low Risk** \nYour digital usage appears healthy! Good job maintaining balance.", st.success),
            1: ("**Moderate Risk** \nYour usage patterns suggest emerging dependency. Consider the recommendations below.", st.warning),
            2: ("**High Risk** \nYour usage indicates potential digital addiction. We recommend taking immediate steps to reduce screen time.", st.error),
        }
        message, message_fn = risk_messages.get(prediction, risk_messages[2])
        message_fn(message)
        risk_score = probability[prediction]

        st.write("")
        st.write("Risk Probability Score")
        st.progress(risk_score, text=f"Confidence: {risk_score*100:.1f}%")

         
        st.subheader("Key Contributing Factors")
        factors = []
        if screen_time > 6: factors.append(f"High Screen Time ({screen_time}h)")
        if social_media_time > 2: factors.append(f"High Social Media Use ({social_media_time}h)")
        if sleep_hours < 7: factors.append(f"Low Sleep Duration ({sleep_hours}h)")
        if phone_pickups > 60: factors.append(f"Frequent Phone Unlocks ({phone_pickups})")
        
        if factors:
            st.write("The following indicators may be raising your risk level:")
            for factor in factors:
                st.markdown(f"- **{factor}**")

        st.subheader("Usage Comparison")
        comparison_df = pd.DataFrame({
            'Hours': [screen_time, 2, 7],
            'Benchmark': ['Your Screen Time', 'Healthy Limit (Rec)', 'Global Average(Screen Time)']
        }).set_index('Benchmark')
        
        st.bar_chart(comparison_df)

        st.subheader("Personalized Recommendations")
        
        recommendations = []
        if screen_time > 8:
            recommendations.append("Try to reduce daily screen time to under 6 hours")
        
        if social_media_time > 3:
            recommendations.append("Limit social media to 2 hours per day")
        
        if sleep_hours < 7:
            recommendations.append("Aim for 7-9 hours of sleep per night")
        
        if phone_pickups > 80:
            recommendations.append("Use 'Do Not Disturb' mode more frequently")
        
        if notifications > 150:
            recommendations.append("Disable non-essential notifications")
        
        if gaming_time > 3:
            recommendations.append("Set time limits for gaming sessions")
            
        if recommendations:
            for rec in recommendations:
                st.info(f"• {rec}")
        else:
            st.success("Great job! Your digital habits appear balanced. Keep it up!")
        
       
        with st.expander("Additional Resources"):
            st.markdown("""
            1) Digital Wellbeing Tools
            - **[Forest](https://www.forestapp.cc/)**: Gamify your focus time.
            - **[Freedom](https://freedom.to/)**: Block distracting websites and apps.
            - **[Google Digital Wellbeing](https://wellbeing.google/)**: Understand your tech usage habits.

            2) Professional Support
            - **[Find A Helpline](https://findahelpline.com/)**: Free, confidential support from a helpline near you.
            - **Note**: If you feel overwhelmed, please speak to a mental health professional.

            3) Actionable Tips
            - **The 20-20-20 Rule**: Every 20 minutes, look at something 20 feet away for 20 seconds.
            - **Phone-Free Zones**: Keep devices out of the bedroom to improve sleep quality.
            """
)
if __name__ == "__main__":
    main()

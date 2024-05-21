import gc
import streamlit as st
import time
from joblib import load

import pandas as pd

# Load the model
@st.cache_resource 
def load_model():
    primary_path = 'models/best_model_bank_marketing.joblib'
    
    try:
        return load(primary_path)
    except FileNotFoundError:
        raise Exception("Model not found in both primary and alternative directories!")

# Load the pipeline
@st.cache_resource
def load_pipeline():
    primary_path = 'models/best_pipeline_bank_marketing.joblib'
    
    try:
        return load(primary_path)
    except FileNotFoundError:
        raise Exception("Pipeline not found in both primary and alternative directories!")  

def get_session_value(key, default_value):
    
    if key not in st.session_state:
        st.session_state[key] = default_value
    return st.session_state[key]

def session_slider(label, min_value, max_value, key, default_value):

    value = get_session_value(key, default_value)
    new_value = st.slider(label, min_value, max_value, value)
    st.session_state[key] = new_value
    return new_value

def session_radio(label, options, key, default_value):

    value = get_session_value(key, default_value)
    new_value = st.radio(label, options, index=options.index(value))
    st.session_state[key] = new_value
    return new_value

def session_selectbox(label, options, key, default_value):

    value = get_session_value(key, default_value)
    new_value = st.selectbox(label, options, index=options.index(value))
    st.session_state[key] = new_value
    return new_value

def session_number_input(label, key, default_value, **kwargs):

    value = get_session_value(key, default_value)
    new_value = st.number_input(label, value=value, **kwargs)
    st.session_state[key] = new_value
    return new_value
    
@st.cache_data
def make_prediction(inputs):
    
    optimal_threshold = 0.27
    
    tweak_inputs = load_pipeline().transform(pd.DataFrame([inputs]))

    y_prob = load_model().predict_proba(tweak_inputs)[:,1]
    
    y_pred = (y_prob >= optimal_threshold).astype(int)

    return y_pred[0]

def main():
    
    gc.enable()

    st.title("Are they subscribing to term deposits?")
    st.subheader("Step into the shoes of a financial detective.")
    
    if "client_name" not in st.session_state:
        st.session_state.client_name = "John"
    
    client_name = st.text_input("Enter your name", st.session_state.client_name)
    
    st.session_state.client_name = client_name
    
    with st.form('user_inputs'):

        st.header("Chapter 1: The Mysterious Client")
        st.write("Unveiling the Persona")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = session_slider("Age. Rumor has it, the client is around...?", 18, 100, 'age', 30)
            marital = session_radio("Marital Status. A ring? A heartbreak? Their marital story is...?", ["married", "divorced", "single"], 'marital', 'single')
        
        with col2:
            education = session_selectbox(
           "Highest Education. Did our client walk the halls of academia?", 
           ["primary", "secondary", "tertiary", "unknown"], 
           'education', 
           'unknown'
           )
            job = session_selectbox(
            "Job. Whispers in the alley mention the client's profession is...?",
            ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services"], 
            'job', 
            'unknown'
            )
    
        st.header("Chapter 2: Secrets of the Wallet")
        st.write("Unlocking Financial Chronicles")
        
        col3, col4 = st.columns(2)
        
        with col3:
            balance = session_number_input("Average Yearly Balance. A little birdie said the average balance was...?", 'balance', 2000)
            default = session_radio("Credit in Default. Any skeletons of past defaults?", ["yes", "no"], 'default', 'no')
            
        with col4:
            housing = session_radio("Housing Loan. The key to a house... or just dreams?", ["yes", "no"], 'housing', 'no')
            loan = session_radio("Personal Loan. Personal tales... or personal loans?", ["yes", "no"], 'loan', 'no')
    
        st.header("Chapter 3: The Last Call")
        st.write("Echoes from the Past")
        
        col5, col6 = st.columns(2)
        
        with col5:
            contact = session_selectbox(
            "Contact Communication Type. A letter? A call? Maybe a message in a bottle?", 
            ["telephone", "cellular", "unknown"], 
            'contact', 
            'unknown'
        )
            day = session_slider("Last Contact Day of the Month. (1-31)", 1, 31, 'day', 15)
            month = session_selectbox("Last Contact Month of the Year.", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 'month', 'jan')
            poutcome = session_selectbox("Outcome of the Previous Marketing Campaign. Were those campaigns a success or failure?", ["unknown", "failure", "success"], 'poutcome', 'unknown')
        
        with col6:
            duration = session_number_input("Last Contact Duration (in seconds). How long was the last conversation?", 'duration', 300)
            campaign = session_number_input("Number of Contacts Performed during this Campaign and for this Client. How many times have voices from the bank echoed?", 'campaign', 1)
            st.write("")
            pdays = session_number_input("Number of Days after the client was last contacted from a previous campaign. (-1 for never)", 'pdays', 30)
            previous = session_number_input("Number of Contacts Performed before this Campaign and for this Client", 'previous', 3)
    
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write(f"Dignosing {client_name}'s deposit outcome...")
            progress_bar = st.progress(0)
            
            for perc_completed in range(100):
                time.sleep(0.01)
                progress_bar.progress(perc_completed+1)
            
            inputs = {
                'age': age,
                'job': job,
                'marital': marital,
                'education': education,
                'balance': balance,
                'default': default,
                'housing': housing,
                'loan': loan,
                'contact': contact,
                'day': day,
                'month': month,
                'duration': duration,
                'campaign': campaign,
                'pdays': pdays,
                'previous': previous,
                'poutcome': poutcome,
            }
            
            prediction = make_prediction(inputs)
            if prediction == 1:
                st.success("Our analysis suggests the client will subscribe to the term deposit.")
            else:
                st.error("Our analysis suggests the client won't subscribe to the term deposit.")
                
            del(
                client_name,
                col1,
                col2,
                col3,
                col4,
                col5,
                col6,
                submitted,
                inputs,
                prediction
            )
            gc.collect()
            
if __name__ == "__main__":
    main()
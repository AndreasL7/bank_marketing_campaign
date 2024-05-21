import streamlit as st
import gc

def main():
    
    gc.enable()
    
    st.title("The Quest for the Golden Client")
    st.subheader("Welcome, Detective!")
    
    st.write("""
    In a world where competition is stiff and every client counts, bank marketers are on a quest to find the **'Golden Client'** - the one most likely to subscribe to a term deposit. In short, a term deposit is ... 
    """)
    
    image_url = "https://www.investopedia.com/thmb/cFsimRVz4FL5B1_HZlg9zahlkQQ=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/termdeposit.asp-Final-dc11a627a6df4ec2ad1d0072973187c1.png"
    
    st.markdown(f'<a href="{image_url}"><img src="{image_url}" alt="description" width="700"/></a>', unsafe_allow_html=True)
    
    st.write("""
    But how do they identify this client amidst the sea of data? As a profit-maximising business, banks have numerous strategies such as running direct marketing campaigns, reaching out to potential clients with an offer they hope is too good to refuse — a term deposit. But so far, the response has been... unpredictable. They've amassed data, a lot of it, but the patterns remain elusive.
    """)
    
    image_url = "https://pbs.twimg.com/media/FNq9-clVUAM0r8V.jpg"
    
    st.subheader("Uncovering the Mystery")
    st.markdown(f'<a href="{image_url}"><img src="{image_url}" alt="description" width="700"/></a>', unsafe_allow_html=True)
    
    st.write("""
    That's where we come in! With our analytical prowess and detective skills to identify patterns, we can help marketers identify whether a client is likely to subscribe to the term deposit. We will try to unveil the hidden patterns in the data and provide invaluable insights.

    Here, we dive deep into the data, build machine learning models to classify clients based on their background and interaction, and try our best to explain the result of our finding. Let's solve this mystery together. The journey promises to be intriguing, challenging, and rewarding!
    
    Navigate to the **Prediction and Modelling** page to experiment with our model, or head over to **Insights from data** to unravel the hidden stories within our training data. For curious minds, we also provide further insights into how our model works behind the scene. 
    
    Grab your coffee and enjoy the investigation ahead! ☕️
    """)
    
    gc.collect()

if __name__ == "__main__":
    main()
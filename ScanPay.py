import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('C:/Users/shriv/PoS/product_classifier.h5')

# Define product prices
product_prices = {
    'Alphenlibe': 10,
    'Mariegold': 20,
    'Mentos': 10
}

# Initialize or reset bill and image in session state
if 'bill_df' not in st.session_state:
    st.session_state.bill_df = pd.DataFrame(columns=["Product", "Quantity", "Price"])
if 'image' not in st.session_state:
    st.session_state.image = None
if 'page' not in st.session_state:
    st.session_state.page = "Home"  # Default to home page

def preprocess_image(image):
    """Preprocess the image for model prediction."""
    image = image.resize((224, 224))  # Resize to the input size expected by your model
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def predict_product(image):
    """Predict product and price using the model."""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    
    # Assuming the model returns class probabilities or a class index
    product_names = list(product_prices.keys())  # List of product names
    predicted_class = np.argmax(predictions)  # Get the predicted class index
    product = product_names[predicted_class]
    price = product_prices[product]
    
    return product, price

def add_to_bill(product, price):
    """Add or update a product in the current bill."""
    # Check if the product is already in the bill
    if st.session_state.bill_df['Product'].str.contains(product).any():
        # Update existing product
        st.session_state.bill_df.loc[st.session_state.bill_df['Product'] == product, 'Quantity'] += 1
        st.session_state.bill_df.loc[st.session_state.bill_df['Product'] == product, 'Price'] = product_prices[product] * st.session_state.bill_df.loc[st.session_state.bill_df['Product'] == product, 'Quantity']
    else:
        # Add new product
        new_entry = pd.DataFrame([{"Product": product, "Quantity": 1, "Price": price}])
        st.session_state.bill_df = pd.concat([st.session_state.bill_df, new_entry], ignore_index=True)

def generate_bill():
    """Generate and store the final bill."""
    if 'bill_df' in st.session_state and not st.session_state.bill_df.empty:
        df = st.session_state.bill_df
        total_price = df['Price'].sum()
        gst = total_price * 0.18  # Example GST calculation
        total_with_gst = total_price + gst
        
        st.session_state.final_bill = {
            'df': df,
            'total_price': total_price,
            'gst': gst,
            'total_with_gst': total_with_gst
        }
        st.session_state.page = "FinalBill"  # Navigate to the final bill page
    else:
        st.session_state.final_bill = None

# Streamlit app
if st.session_state.page == "Home":
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home", "Profile", "Transactions"])

    if selection == "Home":
        st.title("Product Scanner and Billing System")
        
        # Create two columns: one for main content and one for the bill summary
        col1, col2 = st.columns([3, 1])  # Adjust column widths as needed

        with col1:
            # Use Streamlit's camera input widget for capturing images via the browser
            image = st.camera_input("Scan Product")
            
            if image:
                # Convert image to PIL format for prediction
                image_pil = Image.open(image)
                product, price = predict_product(image_pil)
                add_to_bill(product, price)
            
            if st.button("Generate Bill"):
                generate_bill()

        with col2:
            st.write("Current Bill Summary:")
            bill_df = st.session_state.bill_df
            if not bill_df.empty:
                st.write(bill_df)
            else:
                st.write("No products scanned yet.")

    elif selection == "Profile":
        st.title("Profile")
        st.write("This is where user profile information will be displayed.")

    elif selection == "Transactions":
        st.title("Transactions")
        st.write("This is where transaction history will be displayed.")

elif st.session_state.page == "FinalBill":
    st.title("Final Bill")
    
    if 'final_bill' in st.session_state and st.session_state.final_bill:
        df = st.session_state.final_bill['df']
        total_price = st.session_state.final_bill['total_price']
        gst = st.session_state.final_bill['gst']
        total_with_gst = st.session_state.final_bill['total_with_gst']
        
        st.write("Final Bill:")
        st.write(df)
        st.write(f"Total Price: ${total_price:.2f}")
        st.write(f"GST: ${gst:.2f}")
        st.write(f"Total with GST: ${total_with_gst:.2f}")
        
        if st.button("Back to Home"):
            st.session_state.page = "Home"
    else:
        st.write("No bill to display.")

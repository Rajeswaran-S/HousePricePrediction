import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import base64

# Custom CSS for better styling
st.markdown("""
<style>
body {
    background: linear-gradient(45deg, #0a0a2e, #1a1a3a);
    color: #fff;
}

.stApp {
    background: linear-gradient(135deg, rgba(18, 6, 37, 0.9), rgba(24, 11, 56, 0.9));
    border: 1px solid rgba(255, 0, 255, 0.1);
}

.prediction-card {
    background: linear-gradient(180deg, #1e1e3f, #2d1b4e);
    padding: 30px;
    border-radius: 15px;
    margin: 20px 0;
    text-align: center;
    box-shadow: 0 0 20px rgba(123, 0, 255, 0.2),
                0 0 40px rgba(0, 255, 255, 0.1);
    border: 1px solid rgba(123, 0, 255, 0.3);
}

.prediction-card h2 {
    color: #00ffff;
    margin-bottom: 15px;
    text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
}

.price-text {
    color: #ff00ff;
    font-size: 48px;
    font-weight: bold;
    text-shadow: 0 0 15px rgba(255, 0, 255, 0.5);
    padding: 20px 0;
}

.house-card {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 12px;
    margin: 15px 0;
    border: 1px solid rgba(255, 255, 255, 0.2);
    transition: transform 0.2s ease;
}

.house-card:hover {
    transform: scale(1.01);
}

.house-card .price-text {
    color: #00ffff;
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 15px;
}

.house-card .feature-text {
    color: #ffffff;
    font-size: 16px;
    margin: 8px 0;
    line-height: 1.5;
}

.stButton button {
    background: linear-gradient(45deg, #ff00ff, #00ffff);
    color: white;
    border: none;
    text-shadow: 0 0 5px rgba(255, 255, 255, 0.5);
}

.stButton button:hover {
    background: linear-gradient(45deg, #00ffff, #ff00ff);
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
}

/* Add these new styles for the glow effect */
.stNumberInput, .stSelectbox {
    border: 1px solid rgba(0, 255, 255, 0.2) !important;
    border-radius: 8px !important;
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.1) !important;
    transition: all 0.3s ease !important;
}

.stNumberInput:hover, .stSelectbox:hover {
    border-color: rgba(255, 0, 255, 0.4) !important;
    box-shadow: 0 0 15px rgba(255, 0, 255, 0.2) !important;
}

.stNumberInput:focus, .stSelectbox:focus {
    border-color: #00ffff !important;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.3) !important;
}

/* Enhance slider glow */
.stSlider {
    filter: drop-shadow(0 0 10px rgba(0, 255, 255, 0.2));
}

.stSlider > div > div > div {
    background: linear-gradient(90deg, #ff00ff, #00ffff) !important;
}

/* Add glow to tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    padding: 5px;
    box-shadow: 0 0 15px rgba(123, 0, 255, 0.1);
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border-radius: 10px;
    color: #fff;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: rgba(255, 255, 255, 0.1);
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.2);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(45deg, rgba(255, 0, 255, 0.2), rgba(0, 255, 255, 0.2)) !important;
    box-shadow: 0 0 15px rgba(123, 0, 255, 0.3) !important;
}
</style>
""", unsafe_allow_html=True)

# Load and preprocess the data
@st.cache_data
def load_data():
    df = pd.read_csv('Housing.csv')
    return df

def preprocess_data(df):
    # Create a copy of the dataframe
    data = df.copy()
    
    # Create a global LabelEncoder for each categorical column
    global label_encoders
    label_encoders = {}
    categorical_cols = ['mainroad', 'basement', 'furnishingstatus']
    
    # Convert categorical variables to numerical
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])
    
    # Drop columns we're not using
    columns_to_keep = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 
                      'mainroad', 'basement', 'parking', 'furnishingstatus']
    data = data[columns_to_keep]
    
    return data

def train_model(data):
    # Ensure all data is numeric
    X = data.drop('price', axis=1).astype(float)
    y = data['price'].astype(float)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def create_visualizations(df):
    st.subheader("Dataset Insights 📊")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price Distribution
        fig_price = px.histogram(df, x='price', 
                               title='House Price Distribution',
                               color_discrete_sequence=['#00ffff'])
        fig_price.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff'
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Area vs Price Scatter
        fig_scatter = px.scatter(df, x='area', y='price',
                               title='Area vs Price Correlation',
                               color_discrete_sequence=['#ff00ff'])
        fig_scatter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Average Price by Bedrooms
        avg_price_bed = df.groupby('bedrooms')['price'].mean().reset_index()
        fig_beds = px.bar(avg_price_bed, x='bedrooms', y='price',
                         title='Average Price by Bedrooms',
                         color_discrete_sequence=['#00ffff'])
        fig_beds.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff'
        )
        st.plotly_chart(fig_beds, use_container_width=True)
        
        # Furnishing Status Distribution
        fig_furnish = px.pie(df, names='furnishingstatus',
                            title='Furnishing Status Distribution',
                            color_discrete_sequence=['#ff00ff', '#00ffff', '#ff69b4'])
        fig_furnish.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff'
        )
        st.plotly_chart(fig_furnish, use_container_width=True)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/jpg;base64,%s");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-color: rgba(0, 0, 0, 0.7);
        background-blend-mode: overlay;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

def main():
    set_background('c:\\Users\\acer\\Desktop\\HI\\premium_photo-1685148902854-9b9bb49fff08.JPG')
    st.title('🏠 Real Estate Price Prediction and Recommendation System')
    
    # Load and preprocess data
    df = load_data()
    processed_data = preprocess_data(df)
    model = train_model(processed_data)
    
    # Remove the create_visualizations(df) call here
    
    tab1, tab2 = st.tabs(["🔍 Predict Price", "💰 Find Houses"])
    
    with tab1:
        st.header("Predict Your Dream House Price")
        
        col1, col2 = st.columns(2)
        
        with col1:
            area = st.number_input('Area (sq ft)', min_value=1000, max_value=15000, value=3000)
            bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=6, value=2)
            bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=4, value=1)
            stories = st.number_input('Number of Stories', min_value=1, max_value=4, value=1)
        
        with col2:
            mainroad = st.selectbox('Main Road Access', ['yes', 'no'])
            basement = st.selectbox('Basement Available', ['yes', 'no'])
            parking = st.number_input('Parking Spaces', min_value=0, max_value=3, value=1)
            furnishing = st.selectbox('Furnishing Status', ['furnished', 'semi-furnished', 'unfurnished'])
        
        if st.button('Predict Price 🎯'):
            input_data = {
                'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
                'stories': stories, 'mainroad': label_encoders['mainroad'].transform([mainroad])[0],
                'basement': label_encoders['basement'].transform([basement])[0],
                'parking': parking,
                'furnishingstatus': label_encoders['furnishingstatus'].transform([furnishing])[0]
            }
            
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            
            st.markdown(f"""
            <div class="prediction-card">
                <h2>Estimated House Price</h2>
                <p class="price-text">₹{prediction:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.header("Find Your Perfect Home")
        
        col1, col2 = st.columns([2, 1])  # 2:1 ratio for columns
        
        with col1:
            budget = st.slider('Your Budget (₹)', 
                             min_value=1000000, 
                             max_value=15000000, 
                             value=5000000,
                             format="₹%d")
            
            if st.button('Search Houses 🔍'):
                budget_range = (budget * 0.9, budget * 1.1)
                filtered_houses = df[
                    (df['price'] >= budget_range[0]) & 
                    (df['price'] <= budget_range[1])
                ]
                
                if len(filtered_houses) > 0:
                    st.write(f"Found {len(filtered_houses)} houses within your budget range!")
                    
                    for _, house in filtered_houses.iterrows():
                        st.markdown(f"""
                        <div class="house-card">
                            <p class="price-text">₹{house['price']:,.2f}</p>
                            <p class="feature-text">🏗️ {house['area']} sq ft | 🛏️ {house['bedrooms']} Bedrooms | 🚿 {house['bathrooms']} Bathrooms</p>
                            <p class="feature-text">�층 {house['stories']} Stories | 🚗 {house['parking']} Parking Spaces</p>
                            <p class="feature-text">🏠 {house['furnishingstatus']} | 🛣️ Main Road: {house['mainroad']} | 🏗️ Basement: {house['basement']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("No houses found within this budget range. Try adjusting your budget!")
            
            # Price Distribution
            fig_price = px.histogram(df, x='price', 
                                   title='Price Distribution',
                                   color_discrete_sequence=['#00ffff'])
            fig_price.update_layout(
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff',
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Furnishing Status Distribution
            fig_furnish = px.pie(df, names='furnishingstatus',
                                title='Furnishing Types',
                                color_discrete_sequence=['#ff00ff', '#00ffff', '#ff69b4'])
            fig_furnish.update_layout(
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff',
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_furnish, use_container_width=True)

if __name__ == "__main__":
    main()
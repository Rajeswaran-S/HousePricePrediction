### Project Overview
This is a Real Estate Price Prediction and Recommendation System built with Python using Streamlit for the web interface. The system helps users:

1. Predict house prices based on various features
2. Find houses within their budget range
### Key Components 
1. Data Processing
- Dataset : Uses Housing.csv containing 545 house listings with features like:
  
  - Price (target variable)
  - Area (sq ft)
  - Bedrooms/Bathrooms count
  - Stories count
  - Amenities (mainroad, basement, parking)
  - Furnishing status
- Preprocessing :
  
  ```python
  def preprocess_data(df):
      # Converts categorical variables (mainroad, basement, furnishingstatus) to numerical
      # using LabelEncoder
      # Keeps only relevant features for modeling
   ```
2. Machine Learning Model
    - Uses Random Forest Regressor (100 estimators) for price prediction
    - Trained on 80% of data with 20% test split
    - Achieves good performance for this scale of dataset
      
3. Visualizations   
 Creates interactive plots using Plotly:
    - Price distribution histogram
    - Area vs Price scatter plot
    - Average price by bedroom count
    - Furnishing status pie chart 4. User Interface Features
Two Main Tabs :

1. 🔍 Predict Price Tab :   
   - Input form for house features
   - Real-time price prediction
   - Stylish prediction card with glow effects
     
2. 💰 Find Houses Tab :   
   - Budget slider (₹1M-₹15M range)
   - Displays houses within ±10% of budget
   - Attractive house cards with all details
   - Interactive price distribution chart

### Technical Highlights
1. Modern UI :   
   - Gradient backgrounds
   - Glowing input fields
   - Animated hover effects
   - Custom card designs with shadows
     
2. Performance Optimizations :   
   - @st.cache_data decorator for data loading
   - Global LabelEncoders for consistent encoding

3. Visual Enhancements : 
   - Dark theme with neon accents
   - Responsive layout (works on mobile/desktop)
   - Custom CSS styling for all components


### How It Works
1. User selects features or budget
2. System processes inputs through the trained model
3. Returns either:
   - Predicted price (with visual styling)
   - Matching houses (with full details)


### Potential Improvements
1. Add more visualization options
2. Include location-based filtering
3. Implement user accounts to save searches
4. Add comparison feature between houses


### Setup Instructions
1. Install requirements:
```bash
pip install streamlit pandas numpy plotly scikit-learn
 ```
2. Run the app:
```bash
streamlit run house_prediction.py
 ```
This system provides an excellent balance between functionality and visual appeal, making it both useful and engaging for users exploring real estate options.

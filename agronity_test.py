import os
import joblib
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Configuration & Instructions
# -----------------------------------------------------------------------------
# 1. Ensure the following files are in the same folder as this script:
#    - preprocessor.joblib
#    - feasibility_clf.joblib
#    - yield_reg.joblib
#    - sihdatasets.csv
#
# 2. IMPORTANT: This script requires scikit-learn version 1.6.1.
#    If you encounter errors, run this command in your terminal:
#    pip install scikit-learn==1.6.1 --force-reinstall
# -----------------------------------------------------------------------------

# Get the directory where this script is located
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = "sihdatasets.csv"

# --------------------
# Data & Model Loading
# --------------------
def load_data():
    """Loads and combines datasets from the CSV files."""
    try:
        # Get the directory where this script is located
        MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
        file1_path = os.path.join(MODEL_DIR, "sihdatasets.csv")
        file2_path = os.path.join(MODEL_DIR, "corrected_soil_dataset1.csv")
        
        # Load both dataframes
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        
        # Concatenate them into a single dataframe
        combined_df = pd.concat([df1, df2], ignore_index=True)
        
        # Rename a column to match the expected format for your model
        combined_df = combined_df.rename(columns={'Major_Crops': 'crop'})
        
        return combined_df
    except FileNotFoundError as e:
        print(f"Error: Dataset file not found: {e}.")
        print("Make sure both 'sihdatasets.csv' and 'updateddatasets.csv' are in the same folder.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None


def load_models():
    """Loads the pre-trained models from the local directory."""
    try:
        preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
        clf = joblib.load(os.path.join(MODEL_DIR, "feasibility_clf.joblib"))
        reg = joblib.load(os.path.join(MODEL_DIR, "yield_reg.joblib"))
        return preprocessor, clf, reg
    except FileNotFoundError:
        print("Error: Model files not found. Make sure all three .joblib files are in the same folder as this script.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred while loading the models: {e}")
        print("This is likely due to a version mismatch in scikit-learn.")
        return None, None, None

# --------------------
# Analysis Functions
# --------------------
def analyze_feasibility(preprocessor, clf, reg, data_df, crop_type, district, area_size, soil_type):
    """
    Analyzes the feasibility and potential profit using the loaded data.
    """
    # Find the row that matches the user's inputs for district and soil type
    match = data_df[(data_df['District'].str.lower() == district.lower()) & 
                    (data_df['Soil_Type'].str.lower() == soil_type.lower())]

    if match.empty:
        return {
            "feasible": False,
            "reasons": [f"No data found for the combination of '{district}' and '{soil_type}'. Please check your spelling or try a different combination."]
        }
    
    # Extract all the required feature values from the found row
    matched_row = match.iloc[0].to_dict()
    
    # 2. Create the input DataFrame for the model
    input_data = {
        "crop": crop_type.lower(),
        "district": matched_row["District"],
        "soil_type": matched_row["Soil_Type"],
        "area_ha": float(area_size) * 0.4047,  # Convert acres to hectares
        "avg_rain": matched_row["Avg_Rainfall_mm"],
        "temp": matched_row["Avg_Temperature_C"],
        "Fertilizer_Usage_kg_per_ha": matched_row["Fertilizer_Usage_kg_per_ha"],
        "pH_Level": matched_row["pH_Level"],
        "Phosphorus_kg_per_ha": matched_row["Phosphorus_kg_per_ha"],
        "Potassium_kg_per_ha": matched_row["Potassium_kg_per_ha"],
        "Nitrogen_kg_per_ha": matched_row["Nitrogen_kg_per_ha"],
        "Organic_Matter_Percentage": matched_row["Organic_Matter_Percentage"],
        "Electrical_Conductivity_dS_per_m": matched_row["Electrical_Conductivity_dS_per_m"],
        "Cation_Exchange_Capacity_meq_per_100g": matched_row["Cation_Exchange_Capacity_meq_per_100g"],
        "Zinc_ppm": matched_row["Zinc_ppm"],
        "Iron_ppm": matched_row["Iron_ppm"],
        "Manganese_ppm": matched_row["Manganese_ppm"],
        "Copper_ppm": matched_row["Copper_ppm"],
    }
    
    input_df = pd.DataFrame([input_data])
    
    # 3. Transform input and get predictions
    try:
        X_input = preprocessor.transform(input_df)
    except ValueError as e:
        return {
            "feasible": False,
            "reasons": [f"Error during data transformation: {e}. This likely means a new crop, district, or soil type was entered that the model has not seen before."]
        }

    feasibility_prob = clf.predict_proba(X_input)[0, 1]
    is_feasible = bool(clf.predict(X_input)[0])

    if is_feasible:
        # Predict yield and calculate profit
        modal_price_per_quintal = matched_row["Mandi_Price_Rupees_per_kg"]
        expected_yield_tpha = reg.predict(X_input)[0]
        
        # To avoid negative yields from the regressor
        expected_yield_tpha = max(0, expected_yield_tpha)
        
       # More realistic profit calculation

        # Adjustment factors
        MARKETING_LOSSES_FACTOR = 0.90     # ~10% reduction due to transport, commission, wastage
        YIELD_REALIZATION_FACTOR = 0.80    # ~20% reduction due to field losses, pests, etc.
        DEFAULT_COST_PER_HA = 30000        # average cost of cultivation per hectare in INR
        
        # Price per ton after accounting for marketing losses
        price_per_ton = modal_price_per_quintal * 10 * MARKETING_LOSSES_FACTOR
        
        # Effective yield (tons/ha) after accounting for field realities
        effective_yield_tpha = expected_yield_tpha * YIELD_REALIZATION_FACTOR
        
        # Total revenue = effective yield × price × area
        total_revenue = effective_yield_tpha * price_per_ton * input_data["area_ha"]
        
        # Total cost (use provided cost_per_ha if available, else fallback to default)
        cost_per_ha = input_data.get("cost_per_ha", DEFAULT_COST_PER_HA)
        total_cost = cost_per_ha * input_data["area_ha"]
        
        # Final profit
        profit = total_revenue - total_cost
           
        # We use a known high-end yield for percentage calculation.
        max_yield_ref = data_df["Crop_Production_Rate_Yearly"].max() 
        yield_percentage = (expected_yield_tpha / max_yield_ref) * 100
        
        # Future revenue projections (assuming 5% annual growth)
        revenue_1yr = total_revenue * 1.05
        revenue_2yr = total_revenue * (1.05)**2
        
        return {
            "feasible": True,
            "probability": feasibility_prob,
            "expected_yield_tpha": expected_yield_tpha,
            "yield_percentage": yield_percentage,
            "profit_rs": profit,
            "total_revenue_rs": total_revenue,
            "revenue_1yr_rs": revenue_1yr,
            "revenue_2yr_rs": revenue_2yr,
            "mandi_price_rs_per_quintal": modal_price_per_quintal
        }
    else:
        # State reasons for unsuitability (a general message as the model's logic is complex)
        reasons = ["Based on the trained model, the combination of factors is not optimal for this crop in this area."]
        return {
            "feasible": False,
            "reasons": reasons
        }

def analyze_image(data_df, filename):
    """
    Analyzes the image based on filename and returns a static diagnosis.
    """
    # Simple rule-based lookup based on filename
    if "maize" in filename:
        crop_name = "Maize"
    elif "tapioca" in filename:
        crop_name = "Tapioca"
    elif "wheat" in filename:
        crop_name = "Wheat"
    elif "cotton" in filename:
        crop_name = "Cotton"
    elif "groundnut" in filename:
        crop_name = "Groundnut"
    elif "paddy" in filename:
        crop_name = "Paddy"
    elif "tapioca" in filename:
        crop_name = "Tapioca"   
    else:
        return {"status": "error", "message": "The uploaded image is not recognized. Please upload a picture of correct crop."}
    
    # Get data from the combined dataset for the identified crop
    crop_data = data_df[data_df['crop'].str.lower() == crop_name.lower()]
    
    if crop_data.empty:
        return {"status": "error", "message": f"Data for '{crop_name}' not found in the dataset."}
        
    # Get average values for the crop
    avg_production = crop_data['Crop_Production_Rate_Yearly'].mean()
    avg_mandi_price = crop_data['Mandi_Price_Rupees_per_kg'].mean()
    
    return {
        "status": "success",
        "crop": crop_name,
        "avg_production": f"{avg_production:.2f} tons per year",
        "mandi_price": f"{avg_mandi_price:.2f}"
    }

# --------------------
# Main execution block - DO NOT RUN as a standalone script anymore
# This file is now a module for app.py
# --------------------
# if __name__ == "__main__":
#     preprocessor, clf, reg = load_models()
#     data_df = load_data()
#     # The rest of the script is now in app.py

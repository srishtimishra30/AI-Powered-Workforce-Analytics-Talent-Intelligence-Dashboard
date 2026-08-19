import pandas as pd
import json
import os

def load_and_merge_data():
    attrition_df = pd.read_csv('outputs/attrition_predictions.csv')
    skill_gap_df = pd.read_csv('outputs/skill_gap_predictions.csv')
    
    print("--- MERGING DATA ---")
    print("No ID column found. Zipping datasets together side-by-side by row number...")
    
    # Merge side-by-side based on row index
    merged_df = pd.concat([attrition_df, skill_gap_df], axis=1)
    
    # Remove any duplicate columns if they exist
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    
    return merged_df

def generate_recommendation(row, index):
    # Using the exact column name from your screenshot!
    skill_gap = row.get('predicted_skill_gap', 0) 
    
    # Trying the most common ML output names for attrition
    attrition_prob = row.get('Attrition_Probability', row.get('predicted_attrition', row.get('Attrition', 0)))
    
    rec = {
        # Creating a fallback Employee ID based on their row number
        "employee_id": index + 1000, 
        "risk_level": "Low",
        "retention_action": "No immediate action required.",
        "training_action": "Standard development track."
    }
    
    # --- RULE 1: Attrition ---
    if attrition_prob > 0.70 or attrition_prob == 1: 
        rec["risk_level"] = "High"
        rec["retention_action"] = "Critical flight risk. Schedule immediate 1-on-1."
    elif attrition_prob > 0.40:
        rec["risk_level"] = "Medium"
        rec["retention_action"] = "Moderate flight risk. Conduct a stay interview."
        
    # --- RULE 2: Skills (Higher gap = worse) ---
    if skill_gap > 0.6: 
        rec["training_action"] = "Significant skill gap. Enroll in mandatory technical bootcamp."
    elif skill_gap > 0.3:
        rec["training_action"] = "Minor skill gap. Assign a senior mentor."
        
    return rec

def main():
    df = load_and_merge_data()
    recommendations = []
    
    # Process each row, passing the index to act as our Employee ID
    for index, row in df.iterrows():
        recommendations.append(generate_recommendation(row, index))
        
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/recommendations.json', 'w') as f:
        f.write(json.dumps(recommendations, indent=4))
        
    print(f"\n--- SUCCESS ---")
    print(f"Generated {len(recommendations)} unique employee recommendations!")

if __name__ == "__main__":
    main()

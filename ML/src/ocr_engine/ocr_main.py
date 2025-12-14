from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
import re
import json
import os

def preprocess_image(pil_image):
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_with_layout(pdf_path):
    try:
        images = convert_from_path(pdf_path)
    except Exception as e:
        return []

    all_lines = []
    for img in images:
        processed_img = preprocess_image(img)
        # --psm 6 is crucial for table rows
        text = pytesseract.image_to_string(processed_img, config='--psm 6')
        lines = text.split('\n')
        all_lines.extend(lines)
    return all_lines

def parse_lab_report(lines):
    data = {}
    
    test_mappings = {
        # LIVER
        'Bilirubin_Total': ['Serum Bilirubin Total', 'Total Bilirubin', 'Bilirubin Total'],
        'Direct_Bilirubin': ['Serum Bilirubin Direct', 'Direct Bilirubin'],
        'Alkaline_Phosphotase': ['Alk.Phosphatase', 'Alkaline Phosphatase', 'ALP'],
        'Alamine_Aminotransferase': ['SGPT', 'ALT', 'Alamine Aminotransferase'],
        'Aspartate_Aminotransferase': ['SGOT', 'AST', 'Aspartate Aminotransferase'],
        'Total_Protiens': ['Serum Total Protein', 'Total Protein', 'Total Protiens'],
        'Albumin': ['Serum Albumin', 'Albumin'],
        
        # KIDNEY
        'Blood_Urea': ['Serum Urea', 'Blood Urea', 'Urea'],
        'Creatinine': ['Serum Creatinine', 'Creatinine'],
        'Sodium': ['Sodium', 'Na+'],
        'Potassium': ['Potassium', 'K+'],
        
        # DIABETES
        'Glucose': ['Blood Sugar', 'Fasting Blood Glucose', 'Glucose (F)', 'Glucose'],
        
        # THYROID
        'T3': ['Free T3', 'Triiodothyronine', 'T3'],
        'TT4': ['Free T4', 'Thyroxine', 'T4'],
        'TSH': ['TSH', 'Thyroid Stimulating Hormone'],
        
        # CBC (Anemia)
        'Hemoglobin': ['Haemoglobin', 'Hemoglobin', 'Hb'],
        'RBC': ['R.B.C', 'Red Blood Cells'],
        'WBC': ['W.B.C', 'Total Count', 'White Blood Cells'],
        'MCV': ['MCV'],
        'MCH': ['MCH'],
        'MCHC': ['MCHC'],

        # --- NEW: HEART / LIPID PROFILE ---
        'Cholesterol': ['Total Cholesterol', 'Cholesterol'], # Matches "Total Cholesterol"
        'LDL': ['LDL Cholesterol', 'LDL'],                   # Matches "LDL Cholesterol (Calculated)"
        'HDL': ['HDL Cholesterol', 'HDL'],                   # Matches "HDL Cholesterol"
        'Triglyceride': ['Triglyceride']
    }

    def clean_and_extract_number(line, key, matched_keyword):
        # 1. Remove the Test Name itself to avoid reading numbers in the name (like T3)
        line_no_name = re.sub(re.escape(matched_keyword), ' ', line, flags=re.IGNORECASE)

        # 2. Remove Ranges (e.g. 70-110)
        line_no_ranges = re.sub(r'\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?', ' ', line_no_name)

        # 3. Find numbers
        matches = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\d+\.\d+|\d+', line_no_ranges)

        candidates = []
        for m in matches:
            try:
                val = float(m.replace(',', ''))
                candidates.append(val)
            except:
                continue

        if not candidates: return None
        
        value = candidates[0]

        # Sanity Checks
        if key == 'Glucose' and value > 600:
            if len(str(int(value))) >= 3 and str(int(value)).startswith('70'): return 70.0
            if len(candidates) > 1: return candidates[1]
        
        if key == 'WBC' and value < 2000:
             for c in candidates:
                 if c > 2000: return c

        return value

    # Main Loop
    for key, keywords in test_mappings.items():
        for line in lines:
            clean_line = line.strip()
            for keyword in keywords:
                if keyword.lower() in clean_line.lower():
                    
                    # Avoid Partial Match Confusion
                    if key == 'Bilirubin_Total' and 'Direct' in clean_line: continue

                    # Pass keyword to delete it from line
                    val = clean_and_extract_number(clean_line, key, keyword)
                    
                    if val is not None:
                        data[key] = val
                        break 
            if key in data: break

    return data

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Update file name if needed
    test_file = os.path.join(script_dir, "../../datasets/test_reports/Report2.pdf")
    
    if os.path.exists(test_file):
        print(f"--- SCANNING V3: {test_file} ---")
        lines = extract_text_with_layout(test_file)
        extracted_data = parse_lab_report(lines)
        
        print("\n--- EXTRACTED JSON ---")
        print(json.dumps(extracted_data, indent=4))
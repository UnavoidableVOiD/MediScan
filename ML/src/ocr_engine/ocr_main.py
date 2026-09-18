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
        text = pytesseract.image_to_string(processed_img, config='--psm 6')
        lines = text.split('\n')
        all_lines.extend(lines)
    return all_lines

def parse_lab_report(lines):
    data = {}
    
    test_mappings = {
        'Bilirubin_Total': ['Serum Bilirubin Total', 'Total Bilirubin', 'Bilirubin Total'],
        'Direct_Bilirubin': ['Serum Bilirubin Direct', 'Direct Bilirubin'],
        'Alkaline_Phosphotase': ['Alk.Phosphatase', 'Alkaline Phosphatase', 'ALP'],
        'Alamine_Aminotransferase': ['SGPT', 'ALT', 'Alamine Aminotransferase'],
        'Aspartate_Aminotransferase': ['SGOT', 'AST', 'Aspartate Aminotransferase'],
        'Total_Protiens': ['Serum Total Protein', 'Total Protein', 'Total Protiens'],
        'Albumin': ['Serum Albumin', 'Albumin'],
        'Blood_Urea': ['Serum Urea', 'Blood Urea', 'B.Urea'],
        'Creatinine': ['Serum Creatinine', 'Creatinine', 'S.Creatinine'],
        'Sodium': ['Sodium'],
        'Potassium': ['Potassium'],
        'Glucose': ['Blood Sugar', 'Fasting Blood Glucose', 'Glucose (F)', 'Glucose'],
        'T3': ['Free T3', 'Triiodothyronine', 'T3'],
        'TT4': ['Free T4', 'Thyroxine', 'T4'],
        'TSH': ['TSH', 'Thyroid Stimulating Hormone'],
        'Hemoglobin': ['Haemoglobin', 'Hemoglobin', 'Hb'],
        'RBC': ['R.B.C', 'Red Blood Cells'],
        'WBC': ['W.B.C', 'Total Count', 'White Blood Cells'],
        'MCV': ['MCV'],
        'MCH': ['MCH'],
        'MCHC': ['MCHC'],
        'Platelets': ['Platelet Count', 'Platelet', 'PLT', 'Total Platelet Count'],
        'Cholesterol': ['Total Cholesterol', 'Cholesterol'], 
        'LDL': ['LDL Cholesterol', 'LDL'],                   
        'HDL': ['HDL Cholesterol', 'HDL'],                   
        'Triglyceride': ['Triglyceride']
    }

    def clean_and_extract_number(line, key, matched_keyword):
        # 1. Remove the Test Name itself
        line_no_name = re.sub(re.escape(matched_keyword), ' ', line, flags=re.IGNORECASE)

        # 2. FIX: Remove Commas ONLY (Keep spaces to preserve boundaries)
        if key in ['Platelets', 'WBC', 'RBC']:
            line_no_name = line_no_name.replace(',', '')

        # 3. Remove Ranges (Standard format Number-Number)
        line_no_ranges = re.sub(r'\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?', ' ', line_no_name)

        # 4. Find numbers (Integers or Floats)
        matches = re.findall(r'\b\d+(?:\.\d+)?\b', line_no_ranges)

        candidates = []
        for m in matches:
            try:
                val = float(m)
                candidates.append(val)
            except:
                continue

        if not candidates: return None
        
        if key == 'Platelets':
             huge_candidates = [c for c in candidates if c > 50000]
             if huge_candidates: return huge_candidates[0]
             return candidates[-1]

        value = candidates[0]

        if key == 'Glucose' and value > 600:
            if len(str(int(value))) >= 3 and str(int(value)).startswith('70'): return 70.0
            if len(candidates) > 1: return candidates[1]
        
        if key == 'WBC' and value < 2000:
             for c in candidates:
                 if c > 2000: return c

        return value

    for key, keywords in test_mappings.items():
        for line in lines:
            clean_line = line.strip()
            for keyword in keywords:
                if keyword.lower() in clean_line.lower():
                    if key == 'Bilirubin_Total' and 'Direct' in clean_line: continue
                    val = clean_and_extract_number(clean_line, key, keyword)
                    if val is not None:
                        data[key] = val
                        break 
            if key in data: break

    return data

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(script_dir, "../../datasets/test_reports/Report1.pdf")
    if os.path.exists(test_file):
        lines = extract_text_with_layout(test_file)
        print(json.dumps(parse_lab_report(lines), indent=4))
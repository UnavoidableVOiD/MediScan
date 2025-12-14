class SafetyGuard:
    def __init__(self):
        #OCR CORRECTION THRESHOLDS 
        #if value > threshold,we assume OCR missed a decimal point and divide by 10.
        self.fix_thresholds = {
            'Potassium': 15.0,      # e.g., 44 -> 4.4
            'Creatinine': 15.0,     # e.g., 12 -> 1.2
            'Bilirubin_Total': 20.0, # e.g., 15 -> 1.5
            'Hemoglobin': 40.0,     # e.g., 145 -> 14.5
            'TSH': 100.0            # e.g., 250 -> 2.50
        }

        #CRITICAL LIMITS (Panic Values)
        self.critical_limits = {
            'Hemoglobin': {'min': 6.0, 'max': 20.0, 'msg': "CRITICAL: Hemoglobin dangerously abnormal. Risk of heart failure or clotting."},
            'Glucose': {'min': 40.0, 'max': 500.0, 'msg': "CRITICAL: Glucose crisis."},            
            'Potassium': {'min': 2.8, 'max': 6.2, 'msg': "CRITICAL: Potassium imbalance. Immediate cardiac arrest risk."},
            'Sodium': {'min': 120.0, 'max': 160.0, 'msg': "CRITICAL: Severe Sodium imbalance. Risk of seizure or coma."},
            'Creatinine': {'max': 5.0, 'msg': "CRITICAL: Severe Renal Failure indicators."},
            'Bilirubin_Total': {'max': 15.0, 'msg': "CRITICAL: Severe Hyperbilirubinemia (Jaundice). Liver failure risk."},
            'Platelets': {'min': 20000, 'msg': "CRITICAL: Platelets critically low. High risk of internal bleeding."},
            'WBC': {'min': 500, 'max': 50000, 'msg': "CRITICAL: White Blood Cell count indicates severe sepsis or leukemia risk."}
        }

    def sanitize_data(self, data):
        """
        Fixes common OCR decimal errors (e.g. 44 -> 4.4)
        Returns a NEW dictionary with corrected values.
        """
        cleaned_data = data.copy()
        for key, val in cleaned_data.items():
            if key in self.fix_thresholds:
                threshold = self.fix_thresholds[key]
                #if value is insanely high, try dividing by 10
                if val > threshold:
                    fixed_val = val / 10.0
                    #only accept fix if the new value is plausible (less than threshold)
                    if fixed_val < threshold:
                        print(f"SafetyGuard: Auto-Corrected OCR error for {key}: {val} -> {fixed_val}")
                        cleaned_data[key] = fixed_val
        return cleaned_data

    def check_criticals(self, data):
        """
        Scans the extracted OCR data for life-threatening values.
        """
        alerts = []
        
        for key, limits in self.critical_limits.items():
            if key in data and data[key] is not None:
                try:
                    val = float(data[key])
                    
                    if 'max' in limits and val > limits['max']:
                        alerts.append({
                            "parameter": key,
                            "value": val,
                            "limit": f"> {limits['max']}",
                            "message": limits['msg'],
                            "severity": "CRITICAL_HIGH"
                        })
                    
                    if 'min' in limits and val < limits['min']:
                        alerts.append({
                            "parameter": key,
                            "value": val,
                            "limit": f"< {limits['min']}",
                            "message": limits['msg'],
                            "severity": "CRITICAL_LOW"
                        })
                except:
                    continue
                    
        return alerts

if __name__ == "__main__":
    guard = SafetyGuard()
    #test case: OCR error (Potassium 44) + emergency (Glucose 600)
    raw_ocr = {"Glucose": 600, "Hemoglobin": 4.5, "Potassium": 44.0}
    
    print("Raw:", raw_ocr)
    clean_data = guard.sanitize_data(raw_ocr)
    print("Sanitized:", clean_data)
    print("Alerts:", guard.check_criticals(clean_data))
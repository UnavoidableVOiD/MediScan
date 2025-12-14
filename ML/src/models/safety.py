class SafetyGuard:
    def __init__(self):
        # Define CRITICAL LIMITS (Panic Values based on Emergency Medicine Standards)
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
    test_data = {"Glucose": 600, "Hemoglobin": 4.5, "Potassium": 4.0}
    print(guard.check_criticals(test_data))
from fpdf import FPDF
import random
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'NEPAL CENTRAL DIAGNOSTICS - LAB REPORT', 0, 1, 'C')
        self.line(10, 20, 200, 20)
        self.ln(10)

def create_fake_report(filename, disease_type):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    # Patient Info
    pdf.cell(0, 10, f"Patient Name: Synthetic User {random.randint(1,100)}", ln=True)
    pdf.cell(0, 10, f"Age: {random.randint(25, 65)}   Sex: {random.choice(['Male', 'Female'])}", ln=True)
    pdf.ln(10)
    
    # Table Header
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(60, 10, "Test Name", 1)
    pdf.cell(40, 10, "Observed Value", 1)
    pdf.cell(40, 10, "Unit", 1)
    pdf.cell(50, 10, "Reference Range", 1)
    pdf.ln()
    
    # DATA GENERATION BASED ON DISEASE MODELS
    tests = []
    
    if disease_type == "liver":
        tests = [
            ("Total Bilirubin", str(round(random.uniform(0.5, 5.0), 1)), "mg/dL", "0.1 - 1.2"),
            ("Direct Bilirubin", str(round(random.uniform(0.1, 2.0), 1)), "mg/dL", "0.1 - 0.4"),
            ("Alkaline Phosphotase", str(random.randint(150, 500)), "IU/L", "85 - 250"),
            ("Alamine Aminotransferase", str(random.randint(20, 100)), "IU/L", "10 - 40"),
            ("Total Proteins", str(round(random.uniform(5.0, 8.0), 1)), "g/dL", "6.0 - 8.0"),
            ("Albumin", str(round(random.uniform(2.0, 5.0), 1)), "g/dL", "3.5 - 5.5")
        ]
    elif disease_type == "kidney":
        tests = [
            ("Serum Creatinine", str(round(random.uniform(0.5, 5.0), 1)), "mg/dL", "0.6 - 1.2"),
            ("Blood Urea", str(random.randint(20, 100)), "mg/dL", "15 - 45"),
            ("Hemoglobin", str(round(random.uniform(8.0, 16.0), 1)), "g/dL", "12.0 - 16.0"),
            ("Specific Gravity", "1.0" + str(random.randint(10, 30)), "", "1.010 - 1.030"),
            ("Red Blood Cell Count", str(round(random.uniform(3.0, 6.0), 1)), "millions/cmm", "4.5 - 5.5")
        ]
    elif disease_type == "thyroid":
        tests = [
            ("TSH", str(round(random.uniform(0.1, 10.0), 2)), "mIU/L", "0.4 - 4.0"),
            ("T3 (Total)", str(round(random.uniform(0.5, 3.0), 2)), "nmol/L", "1.2 - 2.8"),
            ("T4 (Total)", str(round(random.uniform(50, 160), 1)), "nmol/L", "60 - 150")
        ]
    elif disease_type == "diabetes":
        tests = [
            ("Fasting Blood Glucose", str(random.randint(70, 200)), "mg/dL", "70 - 100"),
            ("Post Prandial Glucose", str(random.randint(100, 250)), "mg/dL", "< 140")
        ]

    # Draw Table
    pdf.set_font("Arial", size=10)
    for name, val, unit, ref in tests:
        pdf.cell(60, 10, name, 1)
        pdf.cell(40, 10, val, 1)
        pdf.cell(40, 10, unit, 1)
        pdf.cell(50, 10, ref, 1)
        pdf.ln()

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    output_dir = os.path.join(script_dir,"../../datasets/test_reports")
    os.makedirs(output_dir, exist_ok=True)
    pdf.output(f"{output_dir}/report_{disease_type}.pdf")
    print(f"Generated: report_{disease_type}.pdf")

# Generate one for each
create_fake_report("test_liver", "liver")
create_fake_report("test_kidney", "kidney")
create_fake_report("test_thyroid", "thyroid")
create_fake_report("test_diabetes", "diabetes")
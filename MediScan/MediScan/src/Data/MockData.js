export const MOCK_DATA = {
  analysis: {
    patientName: "John Doe",
    reportType: "CBC (Complete Blood Count)",
    uploadDate: "Feb 24, 2025",
    summary:
      "The report indicates mild anemia (low hemoglobin) and elevated glucose levels indicating pre-diabetic risk. White blood cell count is normal, suggesting no active infection.",
    doctorSummary:
      "Patient exhibits microcytic anemia with Hemoglobin 11.2 g/dL. Fasting Glucose is 105 mg/dL (Impaired Fasting Glucose). WBC within normal limits. Recommend HbA1c, Iron studies, and lifestyle modification.",
    riskAssessment: [
      { condition: "Anemia", probability: 85, severity: "medium" },
      { condition: "Pre-Diabetes", probability: 62, severity: "medium" },
      { condition: "Infection", probability: 12, severity: "low" }
    ],
    extractedData: [
      { test: "Hemoglobin", value: "11.2", unit: "g/dL", refRange: "13.5 - 17.5", status: "low" },
      { test: "RBC Count", value: "4.1", unit: "mill/mm3", refRange: "4.5 - 5.5", status: "low" },
      { test: "WBC Count", value: "7,500", unit: "/cmm", refRange: "4,000 - 11,000", status: "normal" },
      { test: "Glucose", value: "105", unit: "mg/dL", refRange: "70 - 100", status: "high" },
      { test: "Platelets", value: "250,000", unit: "/cmm", refRange: "150k - 450k", status: "normal" }
    ]
  },
  patients: [
    { id: 1, name: "John Doe", age: 45, lastReport: "Feb 24, 2025", status: "Action Required", type: "CBC" },
    { id: 2, name: "Sarah Smith", age: 32, lastReport: "Feb 20, 2025", status: "Normal", type: "Thyroid" },
    { id: 3, name: "Ram Sharma", age: 58, lastReport: "Feb 18, 2025", status: "Critical", type: "Lipid Profile" }
  ],
  admin: {
    stats: {
      totalUsers: 342,
      totalDoctors: 45,
      pendingDoctors: 3,
      reportsAnalyzed: 1289
    },
    pendingVerifications: [
      {
        id: 101,
        name: "Dr. Anjali Gupta",
        email: "anjali.g@hospital.com",
        license: "NMC-12345",
        specialty: "Cardiology",
        applied: "2 hours ago"
      },
      {
        id: 102,
        name: "Dr. Rajesh K.C.",
        email: "rajesh.kc@clinic.np",
        license: "NMC-88921",
        specialty: "General Medicine",
        applied: "5 hours ago"
      },
      {
        id: 103,
        name: "Dr. Peter Parker",
        email: "p.parker@nyc.med",
        license: "NMC-99999",
        specialty: "Radiology",
        applied: "1 day ago"
      }
    ],
    modelHealth: [
      { name: "OCR Engine", status: "Healthy", accuracy: "98.2%", latency: "1.2s" },
      { name: "Disease Predictor", status: "Training", accuracy: "94.5%", latency: "0.4s" },
      { name: "Chatbot RAG", status: "Healthy", accuracy: "92.1%", latency: "0.8s" }
    ]
  }
};
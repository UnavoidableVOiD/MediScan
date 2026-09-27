/* ------------------------------------------------------------------
   Mock data in the MediScan domain vocabulary (docs/architecture.md).
   Replaced by the generated API client in Phase 2. Shapes mirror the
   intended contracts so pages do not change when the swap happens.
   ------------------------------------------------------------------ */

export type Role = "PATIENT" | "DOCTOR" | "ADMIN";

export type Specialization =
  | "CARDIOLOGIST"
  | "ENDOCRINOLOGIST"
  | "NEPHROLOGIST"
  | "HEPATOLOGIST"
  | "HEMATOLOGIST"
  | "GENERAL_PHYSICIAN";

export interface Doctor {
  id: string;
  name: string;
  specialization: Specialization;
  experience: number;
  fee: number;
  bio: string;
  rating: number;
  reviews: number;
  hospital: string;
  status: "VERIFIED" | "PENDING" | "REJECTED" | "UNVERIFIED";
  nextSlot: string;
}

export interface Observation {
  loinc: string;
  analyte: string;
  panel: "CBC" | "LFT" | "RFT" | "TFT" | "GLUCOSE" | "LIPID";
  value: number;
  unit: string;
  refLow?: number;
  refHigh?: number;
  source: "OCR" | "MANUAL";
  confidence?: number;
  page?: number;
  suggestion?: { proposed: number; reason: string };
}

export type Condition = "ANEMIA" | "CKD" | "LIVER" | "THYROID" | "GLYCAEMIC" | "HEART";

export interface RiskAssessment {
  condition: Condition;
  title: string;
  status: "ASSESSED" | "NOT_ASSESSABLE";
  probability?: number;
  label?: string;
  threshold?: number;
  modelVersion?: string;
  missing?: string[];
  specialist: Specialization;
}

export interface CriticalFlag {
  analyte: string;
  value: number;
  unit: string;
  limit: string;
  severity: "CRITICAL_HIGH" | "CRITICAL_LOW";
  message: string;
  guideline: string;
}

export type JobStatus =
  | "QUEUED"
  | "EXTRACTING"
  | "AWAITING_REVIEW"
  | "EVALUATING"
  | "ASSESSING"
  | "EXPLAINING"
  | "DONE"
  | "FAILED";

export interface Report {
  id: string;
  lab: string;
  uploadedAt: string;
  collectedAt: string;
  status: JobStatus;
  fileName: string;
  pages: number;
  observations: Observation[];
  flags: CriticalFlag[];
  assessments: RiskAssessment[];
  explanation: { patient: string; clinician: string; citations: { doc: string; page: number }[] };
  doctorComment?: { doctor: string; text: string; at: string };
}

export interface Appointment {
  id: string;
  patient: string;
  doctorId: string;
  date: string;
  start: string;
  end: string;
  status: "PENDING" | "PAID" | "COMPLETED" | "CANCELLED";
  amount: number;
  reportId?: string;
}

export interface Patient {
  id: string;
  name: string;
  age: number;
  sex: "M" | "F";
  email: string;
  phone: string;
  city: string;
  lastReport: string;
  risk: "Low" | "Medium" | "High";
}

/* ------------------------------------------------------------ doctors */
export const doctors: Doctor[] = [
  {
    id: "d1",
    name: "Dr. Sunita Karki",
    specialization: "NEPHROLOGIST",
    experience: 12,
    fee: 1200,
    rating: 4.9,
    reviews: 212,
    hospital: "Bir Hospital, Kathmandu",
    status: "VERIFIED",
    nextSlot: "Today, 4:30 PM",
    bio: "Consultant nephrologist focused on early CKD detection and hypertension-related kidney disease.",
  },
  {
    id: "d2",
    name: "Dr. Rajan Shrestha",
    specialization: "ENDOCRINOLOGIST",
    experience: 9,
    fee: 1000,
    rating: 4.8,
    reviews: 168,
    hospital: "Grande International Hospital",
    status: "VERIFIED",
    nextSlot: "Tomorrow, 10:00 AM",
    bio: "Diabetes and thyroid specialist. Believes every patient should understand their own numbers.",
  },
  {
    id: "d3",
    name: "Dr. Anjali Thapa",
    specialization: "HEPATOLOGIST",
    experience: 15,
    fee: 1500,
    rating: 4.9,
    reviews: 301,
    hospital: "Civil Service Hospital",
    status: "VERIFIED",
    nextSlot: "Today, 6:00 PM",
    bio: "Liver disease, fatty liver and hepatitis management with a lifestyle-first approach.",
  },
  {
    id: "d4",
    name: "Dr. Bikash Adhikari",
    specialization: "CARDIOLOGIST",
    experience: 11,
    fee: 1400,
    rating: 4.7,
    reviews: 143,
    hospital: "Shahid Gangalal Heart Centre",
    status: "VERIFIED",
    nextSlot: "Fri, 2:00 PM",
    bio: "Preventive cardiology and lipid management.",
  },
  {
    id: "d5",
    name: "Dr. Priya Maharjan",
    specialization: "HEMATOLOGIST",
    experience: 7,
    fee: 900,
    rating: 4.8,
    reviews: 96,
    hospital: "Patan Hospital",
    status: "VERIFIED",
    nextSlot: "Sat, 11:30 AM",
    bio: "Anemia, iron deficiency and blood disorders in adults and adolescents.",
  },
  {
    id: "d6",
    name: "Dr. Nabin Gurung",
    specialization: "GENERAL_PHYSICIAN",
    experience: 6,
    fee: 700,
    rating: 4.6,
    reviews: 88,
    hospital: "Nepal Medicity",
    status: "PENDING",
    nextSlot: "—",
    bio: "General medicine; awaiting licence verification.",
  },
];

export const specializationLabel: Record<Specialization, string> = {
  CARDIOLOGIST: "Cardiologist",
  ENDOCRINOLOGIST: "Endocrinologist",
  NEPHROLOGIST: "Nephrologist",
  HEPATOLOGIST: "Hepatologist",
  HEMATOLOGIST: "Hematologist",
  GENERAL_PHYSICIAN: "General Physician",
};

/* ------------------------------------------------------------ reports */
const obs = (o: Omit<Observation, "source"> & { source?: Observation["source"] }): Observation => ({
  source: "OCR",
  ...o,
});

export const reports: Report[] = [
  {
    id: "r1042",
    lab: "Bir Hospital — Haematology & Biochemistry",
    uploadedAt: "2026-09-16T09:12:00Z",
    collectedAt: "2026-09-15",
    status: "DONE",
    fileName: "Report_Sep2026.pdf",
    pages: 2,
    observations: [
      obs({
        loinc: "718-7",
        analyte: "Hemoglobin",
        panel: "CBC",
        value: 17.2,
        unit: "g/dL",
        refLow: 13,
        refHigh: 18,
        confidence: 0.98,
        page: 1,
      }),
      obs({
        loinc: "6690-2",
        analyte: "WBC",
        panel: "CBC",
        value: 10400,
        unit: "/cumm",
        refLow: 4000,
        refHigh: 11000,
        confidence: 0.96,
        page: 1,
      }),
      obs({
        loinc: "777-3",
        analyte: "Platelets",
        panel: "CBC",
        value: 258000,
        unit: "/cumm",
        refLow: 150000,
        refHigh: 400000,
        confidence: 0.93,
        page: 1,
      }),
      obs({
        loinc: "787-2",
        analyte: "MCV",
        panel: "CBC",
        value: 87,
        unit: "fL",
        refLow: 80,
        refHigh: 96,
        confidence: 0.97,
        page: 1,
      }),
      obs({
        loinc: "785-6",
        analyte: "MCH",
        panel: "CBC",
        value: 29,
        unit: "pg",
        refLow: 27,
        refHigh: 32,
        confidence: 0.71,
        page: 1,
        suggestion: {
          proposed: 29,
          reason: "OCR read the MCHC line; value re-read from the MCH row",
        },
      }),
      obs({
        loinc: "1558-6",
        analyte: "Fasting glucose",
        panel: "GLUCOSE",
        value: 70,
        unit: "mg/dL",
        refLow: 70,
        refHigh: 110,
        confidence: 0.99,
        page: 1,
      }),
      obs({
        loinc: "1975-2",
        analyte: "Total bilirubin",
        panel: "LFT",
        value: 0.8,
        unit: "mg/dL",
        refLow: 0.3,
        refHigh: 1.0,
        confidence: 0.95,
        page: 1,
      }),
      obs({
        loinc: "1742-6",
        analyte: "ALT (SGPT)",
        panel: "LFT",
        value: 38,
        unit: "IU/L",
        refLow: 5,
        refHigh: 40,
        confidence: 0.94,
        page: 1,
      }),
      obs({
        loinc: "1920-8",
        analyte: "AST (SGOT)",
        panel: "LFT",
        value: 21,
        unit: "IU/L",
        refLow: 5,
        refHigh: 35,
        confidence: 0.94,
        page: 1,
      }),
      obs({
        loinc: "6768-6",
        analyte: "Alkaline phosphatase",
        panel: "LFT",
        value: 144,
        unit: "IU/L",
        refLow: 35,
        refHigh: 150,
        confidence: 0.92,
        page: 1,
      }),
      obs({
        loinc: "2885-2",
        analyte: "Total protein",
        panel: "LFT",
        value: 6.9,
        unit: "g/dL",
        refLow: 6,
        refHigh: 8,
        confidence: 0.64,
        page: 1,
        suggestion: { proposed: 6.9, reason: "Decimal ambiguous in scan (6.0 / 6.9); confirm" },
      }),
      obs({
        loinc: "1751-7",
        analyte: "Albumin",
        panel: "LFT",
        value: 4.0,
        unit: "g/dL",
        refLow: 3.5,
        refHigh: 5.5,
        confidence: 0.97,
        page: 1,
      }),
      obs({
        loinc: "3091-6",
        analyte: "Urea",
        panel: "RFT",
        value: 12,
        unit: "mg/dL",
        refLow: 10,
        refHigh: 45,
        confidence: 0.98,
        page: 1,
      }),
      obs({
        loinc: "2160-0",
        analyte: "Creatinine",
        panel: "RFT",
        value: 0.9,
        unit: "mg/dL",
        refLow: 0.4,
        refHigh: 1.4,
        confidence: 0.99,
        page: 1,
      }),
      obs({
        loinc: "2951-2",
        analyte: "Sodium",
        panel: "RFT",
        value: 135.5,
        unit: "mEq/L",
        refLow: 135,
        refHigh: 145,
        confidence: 0.9,
        page: 1,
      }),
      obs({
        loinc: "2823-3",
        analyte: "Potassium",
        panel: "RFT",
        value: 4.4,
        unit: "mEq/L",
        refLow: 3.5,
        refHigh: 5.2,
        confidence: 0.88,
        page: 1,
      }),
      obs({
        loinc: "3016-3",
        analyte: "TSH",
        panel: "TFT",
        value: 2.24,
        unit: "µIU/mL",
        refLow: 0.35,
        refHigh: 5.5,
        confidence: 0.96,
        page: 2,
      }),
      obs({
        loinc: "3024-7",
        analyte: "Free T4",
        panel: "TFT",
        value: 1.19,
        unit: "ng/dL",
        refLow: 0.8,
        refHigh: 1.8,
        confidence: 0.95,
        page: 2,
      }),
      obs({
        loinc: "2093-3",
        analyte: "Total cholesterol",
        panel: "LIPID",
        value: 172,
        unit: "mg/dL",
        refLow: 75,
        refHigh: 220,
        confidence: 0.98,
        page: 1,
      }),
      obs({
        loinc: "2085-9",
        analyte: "HDL",
        panel: "LIPID",
        value: 31,
        unit: "mg/dL",
        refLow: 35,
        refHigh: 75,
        confidence: 0.97,
        page: 1,
      }),
      obs({
        loinc: "13457-7",
        analyte: "LDL",
        panel: "LIPID",
        value: 117,
        unit: "mg/dL",
        refHigh: 150,
        confidence: 0.97,
        page: 1,
      }),
    ],
    flags: [],
    assessments: [
      {
        condition: "ANEMIA",
        title: "Anemia",
        status: "ASSESSED",
        probability: 0.03,
        label: "Not detected",
        threshold: 0.5,
        modelVersion: "anemia@2026.09.1",
        specialist: "HEMATOLOGIST",
      },
      {
        condition: "CKD",
        title: "Chronic kidney disease",
        status: "ASSESSED",
        probability: 0.06,
        label: "Low risk",
        threshold: 0.35,
        modelVersion: "ckd@2026.09.1",
        specialist: "NEPHROLOGIST",
      },
      {
        condition: "LIVER",
        title: "Liver dysfunction",
        status: "ASSESSED",
        probability: 0.31,
        label: "Borderline",
        threshold: 0.35,
        modelVersion: "liver@2026.09.1",
        specialist: "HEPATOLOGIST",
      },
      {
        condition: "THYROID",
        title: "Thyroid function",
        status: "ASSESSED",
        probability: 0.05,
        label: "Euthyroid",
        threshold: 0.5,
        modelVersion: "thyroid@2026.09.1",
        specialist: "ENDOCRINOLOGIST",
      },
      {
        condition: "GLYCAEMIC",
        title: "Glycaemic risk",
        status: "ASSESSED",
        probability: 0.08,
        label: "Normal fasting glucose",
        threshold: 0.3,
        modelVersion: "rules:ADA-2026",
        specialist: "ENDOCRINOLOGIST",
      },
      {
        condition: "HEART",
        title: "Cardiovascular risk",
        status: "NOT_ASSESSABLE",
        missing: ["Blood pressure", "HbA1c"],
        specialist: "CARDIOLOGIST",
      },
    ],
    explanation: {
      patient:
        "Your blood counts, kidney and thyroid values are all inside their normal ranges. Two things are worth a conversation with a doctor: your HDL (the 'good' cholesterol) is a little low at 31 mg/dL, and your alkaline phosphatase is at the top of its range. Neither is an emergency. Cardiovascular risk could not be assessed because blood pressure and HbA1c were not on this report.",
      clinician:
        "Clinical impression: unremarkable CBC, RFT and TFT. LFT within limits except ALP 144 IU/L (upper-normal) with normal bilirubin and transaminases — consider repeat with GGT. Lipids: HDL 31 mg/dL (low), LDL 117, TC 172. Liver model probability 0.31 (threshold 0.35). CV risk not assessable: BP and HbA1c absent.",
      citations: [
        { doc: "WHO Diabetes Guidelines", page: 14 },
        { doc: "ACG Liver Guideline", page: 22 },
      ],
    },
    doctorComment: {
      doctor: "Dr. Anjali Thapa",
      text: "Agree with the summary. Low HDL — increase aerobic exercise, we can recheck lipids in 3 months. No action needed on ALP.",
      at: "2026-09-17T08:30:00Z",
    },
  },
  {
    id: "r1039",
    lab: "Civil Service Hospital",
    uploadedAt: "2026-09-10T14:03:00Z",
    collectedAt: "2026-09-09",
    status: "DONE",
    fileName: "CSH_Report.pdf",
    pages: 2,
    observations: [
      obs({
        loinc: "718-7",
        analyte: "Hemoglobin",
        panel: "CBC",
        value: 15.7,
        unit: "g/dL",
        refLow: 14,
        refHigh: 18,
        confidence: 0.9,
        page: 1,
      }),
      obs({
        loinc: "777-3",
        analyte: "Platelets",
        panel: "CBC",
        value: 84000,
        unit: "/cumm",
        refLow: 150000,
        refHigh: 400000,
        confidence: 0.95,
        page: 1,
      }),
      obs({
        loinc: "2160-0",
        analyte: "Creatinine",
        panel: "RFT",
        value: 0.9,
        unit: "mg/dL",
        refLow: 0.4,
        refHigh: 1.4,
        confidence: 0.58,
        page: 2,
        source: "MANUAL",
      }),
      obs({
        loinc: "3091-6",
        analyte: "Urea",
        panel: "RFT",
        value: 25,
        unit: "mg/dL",
        refLow: 8,
        refHigh: 45,
        confidence: 0.97,
        page: 2,
      }),
      obs({
        loinc: "2823-3",
        analyte: "Potassium",
        panel: "RFT",
        value: 4.4,
        unit: "mEq/L",
        refLow: 3.5,
        refHigh: 5.2,
        confidence: 0.81,
        page: 2,
      }),
      obs({
        loinc: "3016-3",
        analyte: "TSH",
        panel: "TFT",
        value: 2.51,
        unit: "µIU/mL",
        refLow: 0.35,
        refHigh: 5.5,
        confidence: 0.96,
        page: 2,
      }),
    ],
    flags: [
      {
        analyte: "Platelets",
        value: 84000,
        unit: "/cumm",
        limit: "< 100,000",
        severity: "CRITICAL_LOW",
        message: "Platelet count is critically low. Risk of bleeding — seek medical review today.",
        guideline: "Local critical-value policy",
      },
    ],
    assessments: [
      {
        condition: "ANEMIA",
        title: "Anemia",
        status: "ASSESSED",
        probability: 0.04,
        label: "Not detected",
        threshold: 0.5,
        modelVersion: "anemia@2026.09.1",
        specialist: "HEMATOLOGIST",
      },
      {
        condition: "CKD",
        title: "Chronic kidney disease",
        status: "ASSESSED",
        probability: 0.07,
        label: "Low risk",
        threshold: 0.35,
        modelVersion: "ckd@2026.09.1",
        specialist: "NEPHROLOGIST",
      },
      {
        condition: "THYROID",
        title: "Thyroid function",
        status: "NOT_ASSESSABLE",
        missing: ["Free T4 or Total T4"],
        specialist: "ENDOCRINOLOGIST",
      },
      {
        condition: "LIVER",
        title: "Liver dysfunction",
        status: "NOT_ASSESSABLE",
        missing: ["Bilirubin", "ALT", "AST", "Albumin"],
        specialist: "HEPATOLOGIST",
      },
      {
        condition: "GLYCAEMIC",
        title: "Glycaemic risk",
        status: "NOT_ASSESSABLE",
        missing: ["Fasting glucose"],
        specialist: "ENDOCRINOLOGIST",
      },
      {
        condition: "HEART",
        title: "Cardiovascular risk",
        status: "NOT_ASSESSABLE",
        missing: ["Lipid profile", "Blood pressure"],
        specialist: "CARDIOLOGIST",
      },
    ],
    explanation: {
      patient:
        "URGENT: your platelet count (84,000) is well below the normal range. Platelets help your blood clot, so please see a doctor today. Your kidney and thyroid values look normal. Several conditions could not be assessed because this report only contains a small set of tests.",
      clinician:
        "Clinical impression: isolated thrombocytopenia (84k). RFT and TSH unremarkable. Recommend repeat CBC with peripheral smear; assess for dengue/viral, medication and hepatic causes. Remaining models not assessable on this panel.",
      citations: [{ doc: "WHO Anemia Guideline", page: 9 }],
    },
  },
  {
    id: "r1051",
    lab: "Nepal Medicity",
    uploadedAt: "2026-09-18T07:45:00Z",
    collectedAt: "2026-09-18",
    status: "AWAITING_REVIEW",
    fileName: "photo_report.jpg",
    pages: 1,
    observations: [
      obs({
        loinc: "718-7",
        analyte: "Hemoglobin",
        panel: "CBC",
        value: 11.2,
        unit: "g/dL",
        refLow: 12,
        refHigh: 16,
        confidence: 0.91,
        page: 1,
      }),
      obs({
        loinc: "787-2",
        analyte: "MCV",
        panel: "CBC",
        value: 74,
        unit: "fL",
        refLow: 80,
        refHigh: 96,
        confidence: 0.88,
        page: 1,
      }),
      obs({
        loinc: "2160-0",
        analyte: "Creatinine",
        panel: "RFT",
        value: 12,
        unit: "mg/dL",
        refLow: 0.6,
        refHigh: 1.2,
        confidence: 0.42,
        page: 1,
        suggestion: {
          proposed: 1.2,
          reason: "Value is 10x above the printed range; decimal likely missed",
        },
      }),
      obs({
        loinc: "1558-6",
        analyte: "Fasting glucose",
        panel: "GLUCOSE",
        value: 132,
        unit: "mg/dL",
        refLow: 70,
        refHigh: 100,
        confidence: 0.97,
        page: 1,
      }),
    ],
    flags: [],
    assessments: [],
    explanation: { patient: "", clinician: "", citations: [] },
  },
];

/* ------------------------------------------------------- appointments */
export const appointments: Appointment[] = [
  {
    id: "a1",
    patient: "Prabhat Acharya",
    doctorId: "d3",
    date: "2026-09-19",
    start: "18:00",
    end: "18:20",
    status: "PAID",
    amount: 1500,
    reportId: "r1042",
  },
  {
    id: "a2",
    patient: "Sita Lama",
    doctorId: "d1",
    date: "2026-09-19",
    start: "16:30",
    end: "16:50",
    status: "PAID",
    amount: 1200,
    reportId: "r1039",
  },
  {
    id: "a3",
    patient: "Ramesh Yadav",
    doctorId: "d1",
    date: "2026-09-20",
    start: "10:00",
    end: "10:20",
    status: "PENDING",
    amount: 1200,
  },
  {
    id: "a4",
    patient: "Maya Tamang",
    doctorId: "d2",
    date: "2026-09-12",
    start: "10:00",
    end: "10:20",
    status: "COMPLETED",
    amount: 1000,
  },
  {
    id: "a5",
    patient: "Kiran Bista",
    doctorId: "d1",
    date: "2026-09-11",
    start: "17:00",
    end: "17:20",
    status: "CANCELLED",
    amount: 1200,
  },
];

/* ------------------------------------------------------------ patients */
export const patients: Patient[] = [
  {
    id: "p1",
    name: "Prabhat Acharya",
    age: 22,
    sex: "M",
    email: "prabhat@example.com",
    phone: "+977 98••••1234",
    city: "Kathmandu",
    lastReport: "2026-09-16",
    risk: "Low",
  },
  {
    id: "p2",
    name: "Sita Lama",
    age: 60,
    sex: "F",
    email: "sita@example.com",
    phone: "+977 98••••5678",
    city: "Kavrepalanchok",
    lastReport: "2026-09-10",
    risk: "High",
  },
  {
    id: "p3",
    name: "Ramesh Yadav",
    age: 47,
    sex: "M",
    email: "ramesh@example.com",
    phone: "+977 98••••9012",
    city: "Birgunj",
    lastReport: "2026-08-30",
    risk: "Medium",
  },
  {
    id: "p4",
    name: "Maya Tamang",
    age: 34,
    sex: "F",
    email: "maya@example.com",
    phone: "+977 98••••3456",
    city: "Pokhara",
    lastReport: "2026-09-12",
    risk: "Low",
  },
  {
    id: "p5",
    name: "Kiran Bista",
    age: 51,
    sex: "M",
    email: "kiran@example.com",
    phone: "+977 98••••7890",
    city: "Lalitpur",
    lastReport: "2026-09-01",
    risk: "Medium",
  },
];

/* --------------------------------------------------------------- trends */
export const trends = [
  { date: "Mar 26", Hemoglobin: 15.9, Creatinine: 1.0, "Fasting glucose": 92, LDL: 131 },
  { date: "May 26", Hemoglobin: 16.4, Creatinine: 0.9, "Fasting glucose": 88, LDL: 126 },
  { date: "Jul 26", Hemoglobin: 16.8, Creatinine: 1.0, "Fasting glucose": 81, LDL: 122 },
  { date: "Sep 26", Hemoglobin: 17.2, Creatinine: 0.9, "Fasting glucose": 70, LDL: 117 },
];

/* ---------------------------------------------------------------- admin */
export const adminStats = {
  patients: 1284,
  doctors: 46,
  pendingLicences: 5,
  reportsThisMonth: 3120,
  revenue: 1_236_400,
  commission: 309_100,
  refunds: 36_000,
  perDay: [42, 55, 61, 48, 72, 80, 94, 88, 101, 97, 110, 124, 118, 131],
};

export const conversation = [
  {
    role: "assistant" as const,
    text: "Hello — I've read your Bir Hospital report from 15 Sep. Ask me anything about it.",
  },
  { role: "user" as const, text: "Why is my HDL flagged?" },
  {
    role: "assistant" as const,
    text: "Your HDL is 31 mg/dL and the lab's normal range starts at 35. HDL is the cholesterol that helps clear fat from your arteries, so a low value slightly raises long-term heart risk. It responds well to aerobic exercise and reducing refined carbohydrates. This is not urgent — Dr. Thapa suggested rechecking in three months. Consult a doctor before changing any medication.",
    citations: [{ doc: "ESC Cardiovascular Prevention Guideline", page: 31 }],
  },
];

export const me = {
  name: "Prabhat Acharya",
  email: "prabhat@example.com",
  phone: "+977 9801234567",
  dob: "2004-02-11",
  sex: "M",
  city: "Kathmandu",
  role: "PATIENT" as Role,
};

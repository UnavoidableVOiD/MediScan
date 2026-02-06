import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  FileText,
  CheckCircle2,
  AlertCircle,
  ArrowRight,
  Loader2,
  Search,
  Check,
  FileCheck,
  ClipboardCheck,
  Activity,
  Download,
  ShieldCheck,
} from "lucide-react";

import { toast } from "react-toastify";
import { useDispatch, useSelector } from "react-redux";
import {
  uploadReport,
  processReport,
  correctReportData,
  resetStatus,
  fetchReports,
  clearCurrentReport,
} from "../store/slices/reportsSlice";

import { useNavigate } from "react-router-dom";

const CheckReports = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { uploading, processing, correcting, currentReport, error, success } =
    useSelector((state) => state.reports);

  const [step, setStep] = useState(1);
  const [file, setFile] = useState(null);
  const [extractedData, setExtractedData] = useState({});
  const [analysisStatus, setAnalysisStatus] = useState("idle"); // idle, processing, completed

  const fileInputRef = useRef(null);

  // Unified State Logger
  useEffect(() => {
    console.log("[CheckReports] State Updated:", {
      step,
      currentReport_status: currentReport?.status,
      has_report: !!currentReport,
    });
  }, [step, currentReport]);

  // Sync Redux state with local state when report is uploaded/extracted

  useEffect(() => {
    if (currentReport) {
      if (currentReport.status === "FAILED") {
        toast.error(
          "Report processing failed. Please ensure the file is a valid PDF.",
        );
        setStep(1);
        dispatch(clearCurrentReport());
        return;
      }

      // If we just uploaded or resumed from dashboard, move to Step 2
      if (step === 1 && currentReport.status === "PENDING") {
        setStep(2);
      }

      // If extraction is complete, move to Step 3
      if (
        currentReport.status === "PROCESSED" &&
        currentReport.extracted_data
      ) {
        const data = currentReport.extracted_data.final_data || {};
        setExtractedData(data);

        if (step === 2) {
          setTimeout(() => setStep(3), 500);
        } else if (step === 1) {
          setStep(3);
        }
      }
    }
  }, [currentReport, step]);

  useEffect(() => {
    if (error) {
      toast.error(typeof error === "string" ? error : "An error occurred");
      dispatch(resetStatus());
      if (step === 2) setStep(1); // Go back if upload/extraction fails
    }
  }, [error, dispatch, step]);

  // --- Step 1: File Upload Logic ---
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      validateAndSetFile(selectedFile);
    }
  };

  const validateAndSetFile = (file) => {
    const allowedTypes = ["application/pdf", "image/jpeg", "image/png"];
    const maxSize = 5 * 1024 * 1024; // 5MB

    if (!allowedTypes.includes(file.type)) {
      toast.error("Only PNG files are supported.");
      return;
    }

    if (file.size > maxSize) {
      toast.error("File size must be less than 5MB.");
      return;
    }

    setFile(file);
  };

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    dispatch(uploadReport(formData));
  };

  const handleReset = () => {
    dispatch(clearCurrentReport());
    setFile(null);
    setStep(1);
  };

  const handleProcess = () => {
    console.log("[CheckReports] handleProcess triggered");
    console.log("[CheckReports] Current state before dispatch:", {
      currentReport,
      step,
    });

    if (currentReport && currentReport.id) {
      console.log(
        "[CheckReports] Dispatching processReport for ID:",
        currentReport.id,
      );
      dispatch(processReport(currentReport.id));
    } else {
      console.error(
        "[CheckReports] ABORT: Cannot process. currentReport:",
        currentReport,
      );
      toast.error(
        "Process failed: No active report found. Please try uploading again.",
      );
    }
  };

  // --- Step 2: Verification Logic ---
  const handleVerify = async (e) => {
    e.preventDefault();

    const payload = {
      id: currentReport.id,
      final_data: extractedData,
    };

    setAnalysisStatus("processing");
    const resultAction = await dispatch(correctReportData(payload));

    if (correctReportData.fulfilled.match(resultAction)) {
      setStep(4);
      setAnalysisStatus("completed");
      toast.success("AI Analysis Complete!");
      dispatch(fetchReports()); // Refresh list in background
    } else {
      setAnalysisStatus("idle");
      // Error is handled by the useEffect watching 'error'
    }
  };

  const handleFieldChange = (key, value) => {
    setExtractedData((prev) => ({ ...prev, [key]: value }));
  };

  // --- Helper Components ---
  const StepIndicator = () => {
    const steps = [
      { id: 1, label: "Upload" },
      { id: 2, label: "Extract" },
      { id: 3, label: "Confirm" },
      { id: 4, label: "Analyze" },
    ];

    return (
      <div className="flex items-center justify-center mb-14">
        {steps.map((s, idx) => (
          <div key={s.id} className="flex items-center">
            <div className="flex flex-col items-center relative">
              <div
                className={`w-8 h-8 sm:w-10 sm:h-10 rounded-full flex items-center justify-center font-bold transition-all z-10 ${
                  step === s.id
                    ? "bg-medic-dark text-white ring-4 ring-medic-dark/10 shadow-lg"
                    : step > s.id
                      ? "bg-medic-accent text-medic-dark"
                      : "bg-neutral-soft text-gray-400"
                }`}
              >
                {step > s.id ? (
                  <Check className="w-5 h-5" />
                ) : (
                  <span className="text-sm sm:text-base">{s.id}</span>
                )}
              </div>
              <span
                className={`absolute -bottom-7 text-[9px] sm:text-[10px] font-bold uppercase tracking-wider whitespace-nowrap ${step === s.id ? "text-medic-dark" : "text-gray-400"}`}
              >
                {s.label}
              </span>
            </div>
            {idx < steps.length - 1 && (
              <div
                className={`w-6 sm:w-16 h-1 mx-1 sm:mx-2 rounded-full transition-all ${
                  step > s.id ? "bg-medic-accent" : "bg-neutral-soft"
                }`}
              />
            )}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
      <header className="text-center mb-10 sm:mb-12">
        <h1 className="text-3xl sm:text-4xl font-bold text-medic-dark mb-3 tracking-tight">
          Check Medical Reports
        </h1>
        <p className="text-gray-500 text-base sm:text-lg max-w-lg mx-auto">
          Upload and analyze your medical reports securely with AI.
        </p>
      </header>

      <div className="overflow-x-clip pb-8 sm:pb-0">
        <StepIndicator />
      </div>

      <AnimatePresence mode="wait">
        {/* Step 1: Upload */}
        {step === 1 && (
          <motion.div
            key="step1"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="bg-white rounded-3xl shadow-xl shadow-medic-dark/5 p-8 border border-medic-light/20"
          >
            <div
              className={`border-3 border-dashed rounded-3xl p-12 text-center transition-all ${
                file
                  ? "border-medic-dark bg-medic-light/5"
                  : "border-neutral-soft hover:border-medic-light bg-neutral-soft/50"
              }`}
              onDragOver={(e) => {
                e.preventDefault();
              }}
              onDrop={(e) => {
                e.preventDefault();
                const droppedFile = e.dataTransfer.files[0];
                if (droppedFile) validateAndSetFile(droppedFile);
              }}
            >
              <div className="bg-medic-dark/10 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6">
                <Upload className="w-10 h-10 text-medic-dark" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">
                {file ? file.name : "Drag & drop your report here"}
              </h3>
              <p className="text-gray-500 mb-8 max-w-sm mx-auto">
                Supported formats: PDF: Maximum file size 5MB.
              </p>

              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                className="hidden"
                accept=".pdf"
              />

              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <button
                  onClick={() => fileInputRef.current.click()}
                  className="px-8 py-3 bg-white border-2 border-medic-dark text-medic-dark rounded-xl font-bold hover:bg-medic-light/10 transition-all active:scale-95"
                >
                  Choose File
                </button>
                {file && (
                  <button
                    onClick={handleUpload}
                    disabled={uploading}
                    className="px-8 py-3 bg-medic-dark text-white rounded-xl font-bold shadow-lg shadow-medic-dark/20 hover:bg-medic-primary transition-all active:scale-95 disabled:opacity-50 flex items-center gap-2"
                  >
                    {uploading ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : (
                      "Verify and Upload"
                    )}
                    {!uploading && <ArrowRight className="w-5 h-5" />}
                  </button>
                )}
              </div>
            </div>
          </motion.div>
        )}

        {/* Step 2: Extraction Processing & Manual Trigger */}
        {step === 2 && (
          <motion.div
            key="step2"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.05 }}
            className="bg-white rounded-3xl shadow-xl shadow-medic-dark/5 p-12 border border-medic-light/20 text-center"
          >
            <div className="space-y-8 py-10">
              {processing ? (
                <>
                  <div className="relative w-32 h-32 mx-auto">
                    <div className="absolute inset-0 border-4 border-medic-light/10 rounded-full" />
                    <div className="absolute inset-0 border-4 border-medic-dark rounded-full animate-spin border-t-transparent" />
                    <div className="absolute inset-0 flex items-center justify-center">
                      <Search className="w-12 h-12 text-medic-dark animate-pulse" />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <h2 className="text-2xl font-bold text-gray-900">
                      Extracting Medical Data
                    </h2>
                    <p className="text-gray-500 max-w-sm mx-auto">
                      Our OCR engine is scanning your document for test names,
                      values, and units.
                    </p>
                  </div>
                  <div className="flex items-center justify-center gap-2 text-medic-dark font-medium animate-bounce">
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Processing...</span>
                  </div>
                </>
              ) : (
                <>
                  <div className="bg-medic-light/20 w-32 h-32 rounded-full flex items-center justify-center mx-auto mb-6">
                    <FileText className="w-16 h-16 text-medic-dark" />
                  </div>
                  <div className="space-y-2">
                    <h2 className="text-2xl font-bold text-gray-900">
                      File Uploaded Successfully
                    </h2>
                    <p className="text-gray-500 max-w-sm mx-auto">
                      The report <b>{currentReport?.file.split("/").pop()}</b>{" "}
                      is ready for data extraction. Click the button below to
                      start the OCR process.
                    </p>
                  </div>
                  <div className="pt-6 flex flex-col sm:flex-row items-center justify-center gap-4">
                    <button
                      onClick={handleReset}
                      className="px-8 py-3 bg-white border-2 border-medic-dark text-medic-dark rounded-xl font-bold hover:bg-medic-light/10 transition-all active:scale-95"
                    >
                      Discard and New Upload
                    </button>
                    <button
                      onClick={handleProcess}
                      className="px-10 py-4 bg-medic-dark text-white rounded-2xl font-bold shadow-lg shadow-medic-dark/20 hover:bg-medic-primary transition-all active:scale-95 flex items-center gap-2"
                    >
                      <Activity className="w-5 h-5" />
                      Confirm and Extract Data
                    </button>
                  </div>
                </>
              )}
            </div>
          </motion.div>
        )}

        {/* Step 3: Verification */}
        {step === 3 && (
          <motion.div
            key="step3"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <div className="bg-white rounded-3xl shadow-xl shadow-medic-dark/5 p-8 border border-medic-light/20">
              <div className="flex items-center gap-3 mb-8 text-medic-dark">
                <FileCheck className="w-8 h-8" />
                <h2 className="text-2xl font-bold">Verify Extracted Data</h2>
              </div>

              <form onSubmit={handleVerify} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(extractedData).map(([key, value]) => (
                    <div
                      key={key}
                      className="space-y-1.5 p-4 bg-neutral-soft rounded-2xl border border-transparent hover:border-medic-light/30 transition-all group"
                    >
                      <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">
                        {key.replace(/_/g, " ")}
                      </label>
                      <div className="flex items-center gap-2">
                        <input
                          type="text"
                          value={value || ""}
                          onChange={(e) =>
                            handleFieldChange(key, e.target.value)
                          }
                          className="w-full bg-white px-4 py-2 rounded-xl border border-gray-200 focus:border-medic-dark outline-none transition-all"
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <div className="flex flex-col sm:flex-row items-center justify-end gap-4 pt-6 border-t border-gray-100">
                  <button
                    type="button"
                    onClick={handleReset}
                    className="px-8 py-3 bg-white border-2 border-medic-dark text-medic-dark rounded-xl font-bold hover:bg-medic-light/10 transition-all active:scale-95"
                  >
                    Discard
                  </button>
                  <button
                    type="submit"
                    disabled={correcting}
                    className="px-10 py-4 bg-medic-dark text-white rounded-xl font-bold shadow-lg shadow-medic-dark/20 hover:bg-medic-primary transition-all active:scale-95 disabled:opacity-50 flex items-center gap-2"
                  >
                    {correcting ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : (
                      <ShieldCheck className="w-5 h-5" />
                    )}
                    Confirm & Start AI Analysis
                    {!correcting && <ArrowRight className="w-5 h-5" />}
                  </button>
                </div>
              </form>
            </div>
          </motion.div>
        )}

        {/* Step 4: Analysis Progress & Result */}
        {step === 4 && (
          <motion.div
            key="step4"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-white rounded-3xl shadow-xl shadow-medic-dark/5 p-12 border border-medic-light/20 text-center"
          >
            {analysisStatus === "processing" ? (
              <div className="space-y-8 py-10">
                <div className="relative w-32 h-32 mx-auto">
                  <div className="absolute inset-0 border-4 border-medic-light/30 rounded-full" />
                  <div className="absolute inset-0 border-4 border-medic-dark rounded-full animate-[spin_3s_linear_infinite] border-t-transparent" />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Activity className="w-12 h-12 text-medic-dark animate-pulse" />
                  </div>
                </div>
                <div className="space-y-2">
                  <h2 className="text-2xl font-bold text-gray-900">
                    Analyzing Your Report...
                  </h2>
                  <p className="text-gray-500 max-w-sm mx-auto">
                    Please wait while our AI engine analyzes the results and
                    prepares your health summary.
                  </p>
                </div>

                <div className="max-w-xs mx-auto space-y-4">
                  {[
                    { label: "Uploading secure data", done: true },
                    { label: "Extracting key medical indicators", done: true },
                    { label: "Running AI diagnostic analysis", done: false },
                    { label: "Generating easy-to-read summary", done: false },
                  ].map((s, i) => (
                    <div key={i} className="flex items-center gap-3 text-left">
                      {s.done ? (
                        <CheckCircle2 className="w-5 h-5 text-medic-accent flex-shrink-0" />
                      ) : (
                        <div className="w-5 h-5 rounded-full border-2 border-gray-200 border-t-medic-dark animate-spin flex-shrink-0" />
                      )}
                      <span
                        className={`text-sm font-medium ${s.done ? "text-gray-900" : "text-gray-400"}`}
                      >
                        {s.label}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-8 py-6">
                <div className="bg-medic-accent text-medic-dark w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-6 scale-110">
                  <ClipboardCheck className="w-12 h-12" />
                </div>
                <div className="space-y-2">
                  <h2 className="text-3xl font-bold text-gray-900">
                    Analysis Completed
                  </h2>
                  <p className="text-gray-500">
                    We've successfully analyzed your report. Your insights are
                    ready.
                  </p>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-lg mx-auto pt-6">
                  <button
                    onClick={() =>
                      navigate(`/reports/${currentReport.id}/result/`)
                    }
                    className="flex items-center justify-center gap-2 px-8 py-4 bg-medic-dark text-white rounded-2xl font-bold shadow-lg shadow-medic-dark/20 hover:bg-medic-primary transition-all active:scale-95"
                  >
                    View Detailed Result
                    <ArrowRight className="w-5 h-5" />
                  </button>
                  <button
                    onClick={() => navigate("/dashboard")}
                    className="flex items-center justify-center gap-2 px-8 py-4 bg-white border-2 border-medic-dark text-medic-dark rounded-2xl font-bold hover:bg-medic-light/10 transition-all active:scale-95"
                  >
                    Back to Dashboard
                  </button>
                </div>

                <button
                  onClick={() => setStep(1)}
                  className="text-sm font-bold text-medic-dark hover:underline flex items-center gap-2 mx-auto pt-8"
                >
                  Upload another report
                </button>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      <div className="mt-12 p-6 bg-medic-light/10 rounded-2xl border border-medic-light/20 flex items-start gap-4">
        <AlertCircle className="w-6 h-6 text-medic-dark flex-shrink-0 mt-0.5" />
        <div className="text-sm text-medic-dark/80 leading-relaxed">
          <p className="font-bold mb-1 uppercase tracking-wider text-[10px]">
            Medical Disclaimer
          </p>
          Mediscan AI provides automated report explanations for informational
          purposes only. It is not a clinical diagnosis. Please consult with a
          qualified healthcare professional before making any medical decisions.
        </div>
      </div>
    </div>
  );
};

export default CheckReports;

import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  User,
  FileText,
  Sparkles,
  Download,
  Eye,
  Save,
  CheckCircle2,
  AlertCircle,
  ArrowLeft,
  Clock,
  File,
  Image as ImageIcon,
  Loader2,
  RefreshCcw,
  MessageSquare,
  Stethoscope,
  ShieldCheck,
} from "lucide-react";
import { useSelector, useDispatch } from "react-redux";
import { toast } from "react-toastify";
import {
  fetchMyPatients,
  fetchPatientReports,
  updatePatientNotes,
  submitDoctorComment,
} from "../../store/slices/doctorSlice";

const PatientDetailView = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const { user } = useSelector((state) => state.auth);
  const { patients, currentPatientReports, reportsLoading, notesLoading } =
    useSelector((state) => state.doctor);
  const [loading, setLoading] = useState(true);
  const [doctorNotes, setDoctorNotes] = useState("");
  const [lastSaved, setLastSaved] = useState(null);
  const [patient, setPatient] = useState(null);
  const [selectedReport, setSelectedReport] = useState(null);
  const [commentText, setCommentText] = useState("");
  const [isCommentModalOpen, setIsCommentModalOpen] = useState(false);

  useEffect(() => {
    const load = async () => {
      try {
        // Ensure patients are loaded
        if (patients.length === 0) {
          await dispatch(fetchMyPatients()).unwrap();
        }
        dispatch(fetchPatientReports(id));
      } catch (error) {
        toast.error("Failed to load patient records");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [id, dispatch]);

  // Derive current patient from Redux state
  useEffect(() => {
    if (patients.length > 0) {
      const currentPatient = patients.find((p) => p.id === parseInt(id));
      if (currentPatient) {
        setPatient(currentPatient);
        setDoctorNotes(currentPatient.notes || "");
      } else {
        toast.error("Patient not found");
        navigate("/doctor-dashboard");
      }
    }
  }, [patients, id, navigate]);

  const handleSaveNotes = async () => {
    try {
      await dispatch(
        updatePatientNotes({ patientId: id, notes: doctorNotes }),
      ).unwrap();
      setLastSaved(new Date().toLocaleTimeString());
      setPatient((prev) => ({ ...prev, notes: doctorNotes }));
    } catch (error) {
      // toast handled by slice
    }
  };

  const handleOpenCommentModal = (report) => {
    setSelectedReport(report);
    setCommentText(report.doctor_comment?.comment || "");
    setIsCommentModalOpen(true);
  };

  const handleSaveComment = async () => {
    try {
      await dispatch(
        submitDoctorComment({
          report: selectedReport.id,
          comment: commentText,
        }),
      ).unwrap();
      setIsCommentModalOpen(false);
      // The slice handles updating the report in currentPatientReports
    } catch (error) {
      // toast handled by slice
    }
  };

  if (loading) {
    return (
      <div className="min-h-[calc(100vh-80px)] flex items-center justify-center bg-neutral-background">
        <div className="text-center space-y-4">
          <Loader2 className="w-12 h-12 text-medic-dark animate-spin mx-auto" />
          <p className="text-gray-500 font-bold animate-pulse">
            Retrieving Patient Medical Records...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-[calc(100vh-80px)] bg-neutral-background py-8 px-6">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header / Basic Info */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <button
            onClick={() => navigate("/doctor-dashboard")}
            className="flex items-center gap-2 text-gray-400 hover:text-medic-dark font-bold text-sm transition-colors group"
          >
            <ArrowLeft
              size={18}
              className="group-hover:-translate-x-1 transition-transform"
            />
            Back to Patients
          </button>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-4 py-1.5 bg-green-50 text-green-600 rounded-full text-xs font-black border border-green-100 shadow-sm">
              <ShieldCheck size={14} />
              SECURE ACCESS
            </div>
          </div>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-[2.5rem] shadow-2xl shadow-medic-dark/5 border border-medic-light/20 p-8 md:p-10"
        >
          <div className="flex flex-col md:flex-row gap-10">
            {/* Patient Identity */}
            <div className="w-full md:w-1/3 space-y-8">
              <div className="flex flex-col items-center text-center space-y-4">
                <div className="w-32 h-32 bg-medic-light/20 rounded-3xl flex items-center justify-center overflow-hidden border-4 border-white shadow-xl">
                  <User className="w-16 h-16 text-medic-dark/40" />
                </div>
                <div className="space-y-1">
                  <h1 className="text-2xl font-black text-gray-900 tracking-tight">
                    {patient.first_name} {patient.last_name}
                  </h1>
                  <p className="text-medic-dark font-bold text-sm tracking-wide uppercase">
                    Patient ID: #{patient.id}
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 gap-4">
                <div className="bg-neutral-soft/50 p-4 rounded-2xl border border-gray-50 text-center">
                  <p className="text-[10px] text-gray-400 font-black uppercase tracking-widest mb-1">
                    Status
                  </p>
                  <p
                    className={`font-bold ${patient.status === "ONGOING" ? "text-orange-500" : "text-green-500"}`}
                  >
                    {patient.status}
                  </p>
                </div>
                <div className="bg-neutral-soft/50 p-4 rounded-2xl border border-gray-50">
                  <p className="text-[10px] text-gray-400 font-black uppercase tracking-widest mb-1">
                    Contact Info
                  </p>
                  <p className="font-bold text-gray-900 text-sm truncate">
                    {patient.email}
                  </p>
                  <p className="font-medium text-gray-500 text-xs truncate mt-0.5">
                    {patient.phone_number}
                  </p>
                </div>
                <div className="bg-neutral-soft/50 p-4 rounded-2xl border border-gray-50">
                  <p className="text-[10px] text-gray-400 font-black uppercase tracking-widest mb-1">
                    Latest Risk Level
                  </p>
                  <p className="font-bold text-gray-900">{patient.condition}</p>
                </div>
              </div>
            </div>

            {/* Main Medical Content */}
            <div className="flex-1 space-y-8">
              {/* AI Summary Section */}
              <section className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-medic-dark text-white rounded-xl shadow-lg shadow-medic-dark/20">
                      <Sparkles size={18} />
                    </div>
                    <h2 className="text-xl font-black text-gray-900 tracking-tight">
                      AI Clinical Summary
                    </h2>
                    <span className="px-2 py-0.5 bg-purple-50 text-purple-600 rounded text-[10px] font-black tracking-widest uppercase border border-purple-100">
                      AI Verified
                    </span>
                  </div>
                </div>

                <div className="bg-gradient-to-br from-medic-dark to-[#093d4a] text-white/90 p-8 rounded-[2rem] shadow-xl shadow-medic-dark/20 relative overflow-hidden group">
                  <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full -mr-20 -mt-20 blur-3xl transition-all group-hover:bg-white/10" />
                  <div className="relative space-y-6">
                    {currentPatientReports.length > 0 &&
                    currentPatientReports[0].ai_analysis ? (
                      <>
                        <div>
                          <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-medic-light/60 mb-2">
                            Patient Summary
                          </h4>
                          <p className="text-sm font-medium leading-relaxed">
                            {currentPatientReports[0].ai_analysis.summary}
                          </p>
                        </div>
                        <div>
                          <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-medic-light/60 mb-2">
                            Doctor Insights
                          </h4>
                          <p className="text-sm font-medium leading-relaxed italic">
                            "
                            {
                              currentPatientReports[0].ai_analysis
                                .doctor_summary
                            }
                            "
                          </p>
                        </div>
                        <div className="pt-4 border-t border-white/10">
                          <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-medic-light/60 mb-2">
                            Key Findings
                          </h4>
                          <div className="flex flex-wrap gap-2">
                            {currentPatientReports[0].ai_analysis.key_findings.map(
                              (f, i) => (
                                <span
                                  key={i}
                                  className="px-3 py-1 bg-white/10 rounded-full text-[10px] font-bold"
                                >
                                  {f}
                                </span>
                              ),
                            )}
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="py-8 text-center text-medic-light/40 italic text-sm">
                        No AI analysis available for this patient's reports.
                      </div>
                    )}
                  </div>
                </div>
              </section>

              {/* Reports Grid */}
              <section className="space-y-4 pt-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-neutral-soft text-medic-dark rounded-xl border border-gray-100">
                    <FileText size={18} />
                  </div>
                  <h2 className="text-xl font-black text-gray-900 tracking-tight">
                    Patient Diagnostic Reports
                  </h2>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {currentPatientReports.length > 0 ? (
                    currentPatientReports.map((report) => (
                      <div
                        key={report.id}
                        className="p-5 bg-neutral-soft border border-gray-100 rounded-[1.5rem] flex flex-col gap-4 group hover:bg-white hover:shadow-xl hover:shadow-medic-dark/5 transition-all"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-4">
                            <div className="w-12 h-12 bg-white rounded-2xl flex items-center justify-center text-medic-dark shadow-sm border border-gray-50 group-hover:scale-110 transition-transform">
                              {report.file.endsWith(".pdf") ? (
                                <File size={22} />
                              ) : (
                                <ImageIcon size={22} />
                              )}
                            </div>
                            <div>
                              <h4 className="text-sm font-black text-gray-900 mb-0.5">
                                Report #{report.id}
                              </h4>
                              <p className="text-[10px] text-gray-400 font-bold uppercase tracking-wider">
                                {new Date(
                                  report.uploaded_at,
                                ).toLocaleDateString()}{" "}
                                • {report.status}
                              </p>
                            </div>
                          </div>
                          <div className="flex gap-2">
                            <button
                              onClick={() => handleOpenCommentModal(report)}
                              className="p-2 text-medic-dark hover:bg-medic-light/30 rounded-lg transition-colors"
                              title="Add/Edit Comment"
                            >
                              <MessageSquare size={18} />
                            </button>
                            <a
                              href={report.file}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="p-2 text-gray-400 hover:text-medic-dark transition-colors"
                            >
                              <Eye size={18} />
                            </a>
                          </div>
                        </div>
                        {report.doctor_comment && (
                          <div className="px-4 py-3 bg-white/50 rounded-xl border border-dotted border-medic-dark/10">
                            <p className="text-[10px] text-medic-dark font-black uppercase tracking-widest mb-1">
                              Your Comment
                            </p>
                            <p className="text-xs text-gray-600 italic">
                              "{report.doctor_comment.comment}"
                            </p>
                          </div>
                        )}
                      </div>
                    ))
                  ) : (
                    <div className="col-span-2 py-10 text-center border-2 border-dashed border-gray-100 rounded-[2rem]">
                      <p className="text-gray-400 font-bold italic">
                        No diagnostic reports available.
                      </p>
                    </div>
                  )}
                </div>
              </section>

              {/* Clinical Notes Section */}
              <section className="space-y-4 pt-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-neutral-soft text-medic-dark rounded-xl border border-gray-100">
                      <Stethoscope size={18} />
                    </div>
                    <h2 className="text-xl font-black text-gray-900 tracking-tight">
                      Clinical Observations
                    </h2>
                  </div>
                  {lastSaved && (
                    <div className="flex items-center gap-1.5 text-[10px] font-black text-green-600 uppercase tracking-widest">
                      <CheckCircle2 size={12} />
                      Last Saved {lastSaved}
                    </div>
                  )}
                </div>

                <div className="relative">
                  <textarea
                    value={doctorNotes}
                    onChange={(e) => setDoctorNotes(e.target.value)}
                    placeholder="Add clinical observations, diagnosis notes, or treatment plans..."
                    className="w-full h-48 bg-neutral-soft hover:bg-white focus:bg-white border-2 border-transparent focus:border-medic-dark rounded-[2rem] p-8 outline-none transition-all font-medium text-gray-900 text-sm leading-relaxed shadow-inner placeholder:text-gray-300 placeholder:italic"
                  />
                  <div className="absolute bottom-6 right-6">
                    <button
                      onClick={handleSaveNotes}
                      disabled={
                        notesLoading || doctorNotes === (patient?.notes || "")
                      }
                      className={`flex items-center gap-2 px-8 py-4 rounded-2xl font-black text-sm tracking-wide transition-all shadow-xl active:scale-[0.98] ${
                        doctorNotes === (patient?.notes || "")
                          ? "bg-gray-100 text-gray-400 cursor-not-allowed shadow-none"
                          : "bg-medic-dark text-white hover:bg-medic-primary shadow-medic-dark/20"
                      }`}
                    >
                      {notesLoading ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          SAVING
                        </>
                      ) : (
                        <>
                          <Save size={18} />
                          {patient.notes ? "UPDATE NOTES" : "SAVE OBSERVATIONS"}
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </section>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Comment Modal */}
      <AnimatePresence>
        {isCommentModalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsCommentModalOpen(false)}
              className="absolute inset-0 bg-medic-dark/40 backdrop-blur-sm"
            />
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              className="bg-white rounded-[2.5rem] w-full max-w-lg p-8 shadow-2xl relative z-10 border border-medic-light/20"
            >
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-medic-light/30 text-medic-dark rounded-xl">
                  <MessageSquare size={20} />
                </div>
                <h3 className="text-xl font-black text-gray-900 tracking-tight">
                  Report Feedback
                </h3>
              </div>

              <p className="text-sm text-gray-500 mb-4 font-medium">
                Providing feedback for Report #{selectedReport?.id}
              </p>

              <textarea
                value={commentText}
                onChange={(e) => setCommentText(e.target.value)}
                placeholder="Enter clinical feedback for the patient regarding this report..."
                className="w-full h-40 bg-neutral-soft focus:bg-white border-2 border-transparent focus:border-medic-dark rounded-2xl p-6 outline-none transition-all font-medium text-gray-900 text-sm leading-relaxed"
              />

              <div className="flex gap-4 mt-8">
                <button
                  onClick={() => setIsCommentModalOpen(false)}
                  className="flex-1 py-4 bg-gray-100 text-gray-500 rounded-2xl font-black text-sm tracking-wide hover:bg-gray-200 transition-all"
                >
                  CANCEL
                </button>
                <button
                  onClick={handleSaveComment}
                  disabled={!commentText.trim()}
                  className="flex-2 px-10 py-4 bg-medic-dark text-white rounded-2xl font-black text-sm tracking-wide hover:bg-medic-primary transition-all shadow-xl shadow-medic-dark/20 disabled:opacity-50 disabled:shadow-none"
                >
                  SAVE COMMENT
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default PatientDetailView;

import { useState, useEffect, useMemo } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  User,
  FileText,
  Sparkles,
  Eye,
  Save,
  CheckCircle2,
  ArrowLeft,
  Clock,
  File,
  Download,
  Image as ImageIcon,
  Loader2,
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
  fetchPatientTrends,
  markPatientCompleted,
  updateClinicalObservations,
} from "../../store/slices/doctorSlice";
import { fetchAppointments } from "../../store/slices/appointmentSlice";
import { BASE_URL } from "../../services/api";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const PatientDetailView = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const {
    patients,
    currentPatientReports,
    currentPatientTrends,
    notesLoading,
  } = useSelector((state) => state.doctor);
  const { appointments } = useSelector((state) => state.appointment);
  const [loading, setLoading] = useState(true);
  const [doctorNotes, setDoctorNotes] = useState("");
  const [observations, setObservations] = useState("");
  const [lastSaved, setLastSaved] = useState(null);
  const [patient, setPatient] = useState(null);
  const [selectedReport, setSelectedReport] = useState(null);
  const [commentText, setCommentText] = useState("");
  const [isCommentModalOpen, setIsCommentModalOpen] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState("");

  useEffect(() => {
    const load = async () => {
      try {
        // Ensure patients are loaded
        if (patients.length === 0) {
          await dispatch(fetchMyPatients()).unwrap();
        }
        if (appointments.length === 0) {
          dispatch(fetchAppointments());
        }
        dispatch(fetchPatientReports(id));
        dispatch(fetchPatientTrends(id));
      } catch (error) {
        toast.error("Failed to load patient records");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [id, dispatch, appointments.length, patients.length]);

  // Derive current patient from Redux state
  useEffect(() => {
    if (patients.length > 0) {
      const currentPatient = patients.find((p) => p.id === parseInt(id));
      if (currentPatient) {
        setPatient(currentPatient);
        setDoctorNotes(currentPatient.notes || "");
        setObservations(currentPatient.clinical_observations || "");
      } else {
        toast.error("Patient not found");
        navigate("/doctor-dashboard");
      }
    }
  }, [patients, id, navigate]);

  const availableMetrics = useMemo(() => {
    const metricsSet = new Set();
    currentPatientTrends.forEach((point) => {
      Object.keys(point.metrics).forEach((m) => metricsSet.add(m));
    });
    const metrics = Array.from(metricsSet);
    if (metrics.length > 0 && !selectedMetric) {
      setSelectedMetric(metrics[0]);
    }
    return metrics;
  }, [currentPatientTrends, selectedMetric]);

  const chartData = useMemo(() => {
    return currentPatientTrends
      .filter((point) => point.metrics[selectedMetric] !== undefined)
      .map((point) => ({
        date: point.date,
        value: point.metrics[selectedMetric],
      }));
  }, [currentPatientTrends, selectedMetric]);

  const handleSaveNotes = async () => {
    try {
      await dispatch(
        updatePatientNotes({ patientId: id, notes: doctorNotes }),
      ).unwrap();
      setLastSaved(new Date().toLocaleTimeString());
      setPatient((prev) => ({ ...prev, notes: doctorNotes }));
    } catch (error) {
      console.error("Failed to save notes:", error);
    }
  };

  const handleSaveObservations = async () => {
    try {
      await dispatch(
        updateClinicalObservations({ patientId: id, observations }),
      ).unwrap();
      setLastSaved(new Date().toLocaleTimeString());
      setPatient((prev) => ({ ...prev, clinical_observations: observations }));
    } catch (error) {
      console.error("Failed to save observations:", error);
    }
  };

  const handleMarkCompleted = async () => {
    try {
      await dispatch(markPatientCompleted(id)).unwrap();
      // Re-fetch to ensure all statuses (including appointments) are synced
      dispatch(fetchMyPatients());
      dispatch(fetchAppointments());
      toast.success("Patient session completed");
    } catch (error) {
      console.error("Failed to mark completed:", error);
      toast.error("Failed to update status");
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
                    className={`font-black text-sm tracking-widest uppercase ${patient.status === "ONGOING" ? "text-orange-500" : "text-green-500"}`}
                  >
                    {patient.status}
                  </p>
                  {patient.status === "ONGOING" &&
                    (() => {
                      const today = new Date().toISOString().split("T")[0];
                      // Find the appointment for this patient (assuming 1 active appointment for simplicity or taking the latest)
                      // In reality, we might need to filter by status too.
                      // Here we look for any appointment for this patient that is ONGOING/PAID to check the date.
                      // Since we don't have patient_id directly linked easily sometimes, we use email if possible or ID if available.
                      // DoctorDashboard uses patient ID. Let's assume patient.id is the key.
                      const patientAppt = appointments.find(
                        (a) =>
                          a.patient === patient.id &&
                          (a.status === "PAID" || a.status === "ONGOING"),
                      );

                      const isFuture = patientAppt
                        ? patientAppt.appointment_date > today
                        : false;
                      // If no appointment found, maybe we shouldn't block, or maybe we should?
                      // Let's assume if no appointment found, we allow it (fallback) or disallow.
                      // Safer to allow but show warning? Or disallow?
                      // The prompt says "safeguard till appointment date".
                      // If future, disable.

                      return (
                        <div className="mt-2">
                          <button
                            onClick={handleMarkCompleted}
                            disabled={isFuture}
                            className={`w-full py-2 rounded-xl text-[10px] font-black tracking-widest uppercase border transition-all ${
                              isFuture
                                ? "bg-gray-100 text-gray-400 border-gray-200 cursor-not-allowed"
                                : "bg-green-50 text-green-600 border-green-200 hover:bg-green-100"
                            }`}
                          >
                            {isFuture
                              ? `Wait until ${patientAppt?.appointment_date}`
                              : "MARK COMPLETED"}
                          </button>
                          {isFuture && (
                            <p className="text-[10px] text-center text-gray-400 mt-1">
                              Appointment is in the future.
                            </p>
                          )}
                        </div>
                      );
                    })()}
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
                      Patient Progress & Analysis
                    </h2>
                  </div>
                </div>

                {/* Trends Chart */}
                {availableMetrics.length > 0 && (
                  <div className="bg-white p-6 rounded-[2rem] border border-gray-100 shadow-sm space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-black text-gray-400 uppercase tracking-widest">
                        OCR Value Trends
                      </h3>
                      <select
                        value={selectedMetric}
                        onChange={(e) => setSelectedMetric(e.target.value)}
                        className="bg-neutral-soft px-3 py-1.5 rounded-xl text-xs font-bold border-none outline-none focus:ring-2 focus:ring-medic-dark/20"
                      >
                        {availableMetrics.map((m) => (
                          <option key={m} value={m}>
                            {m}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="h-[250px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData}>
                          <CartesianGrid
                            strokeDasharray="3 3"
                            vertical={false}
                            stroke="#f0f0f0"
                          />
                          <XAxis
                            dataKey="date"
                            axisLine={false}
                            tickLine={false}
                            tick={{
                              fontSize: 10,
                              fontWeight: 700,
                              fill: "#999",
                            }}
                            dy={10}
                          />
                          <YAxis
                            axisLine={false}
                            tickLine={false}
                            tick={{
                              fontSize: 10,
                              fontWeight: 700,
                              fill: "#999",
                            }}
                          />
                          <Tooltip
                            contentStyle={{
                              borderRadius: "1rem",
                              border: "none",
                              boxShadow: "0 10px 25px rgba(0,0,0,0.1)",
                              fontSize: "12px",
                              fontWeight: "bold",
                            }}
                          />
                          <Line
                            type="monotone"
                            dataKey="value"
                            name={selectedMetric}
                            stroke="#1F7A5B"
                            strokeWidth={3}
                            dot={{ r: 4, fill: "#1F7A5B", strokeWidth: 0 }}
                            activeDot={{ r: 6, strokeWidth: 0 }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                <div className="bg-gradient-to-br from-medic-dark to-[#093d4a] text-white/90 p-8 rounded-[2rem] shadow-xl shadow-medic-dark/20 relative overflow-hidden group">
                  <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full -mr-20 -mt-20 blur-3xl transition-all group-hover:bg-white/10" />
                  <div className="relative space-y-6">
                    {currentPatientReports.length > 0 &&
                    currentPatientReports[0].result ? (
                      <>
                        <div>
                          <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-medic-light/60 mb-2">
                            Doctor Insights
                          </h4>
                          <p className="text-sm font-medium leading-relaxed italic">
                            &quot;{currentPatientReports[0].result.summary}
                            &quot;
                          </p>
                        </div>
                        <div className="pt-4 border-t border-white/10">
                          <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-medic-light/60 mb-2">
                            Key Findings
                          </h4>
                          <div className="flex flex-wrap gap-2">
                            {currentPatientReports[0].result.key_findings.map(
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
                        No AI analysis available for this patient&apos;s
                        reports.
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
                              href={`${BASE_URL}${report.file}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="p-2 text-gray-400 hover:text-medic-dark transition-colors"
                              title="View Report"
                            >
                              <Eye size={18} />
                            </a>
                            <a
                              href={`${BASE_URL}${report.file}`}
                              download
                              target="_blank"
                              rel="noopener noreferrer"
                              className="p-2 text-gray-400 hover:text-medic-dark transition-colors"
                              title="Download Report"
                            >
                              <Download size={18} />
                            </a>
                          </div>
                        </div>
                        {report.doctor_comment && (
                          <div className="px-4 py-3 bg-white/50 rounded-xl border border-dotted border-medic-dark/10">
                            <p className="text-[10px] text-medic-dark font-black uppercase tracking-widest mb-1">
                              Your Comment
                            </p>
                            <p className="text-xs text-gray-600 italic">
                              &quot;{report.doctor_comment.comment}&quot;
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
                    value={observations}
                    onChange={(e) => setObservations(e.target.value)}
                    placeholder="Add clinical observations, diagnosis notes, or treatment plans..."
                    className="w-full h-48 bg-neutral-soft hover:bg-white focus:bg-white border-2 border-transparent focus:border-medic-dark rounded-[2rem] p-8 outline-none transition-all font-medium text-gray-900 text-sm leading-relaxed shadow-inner placeholder:text-gray-300 placeholder:italic"
                  />
                  <div className="absolute bottom-6 right-6">
                    <button
                      onClick={handleSaveObservations}
                      disabled={
                        notesLoading ||
                        observations === (patient?.clinical_observations || "")
                      }
                      className={`flex items-center gap-2 px-8 py-4 rounded-2xl font-black text-sm tracking-wide transition-all shadow-xl active:scale-[0.98] ${
                        observations === (patient?.clinical_observations || "")
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
                          {patient?.clinical_observations
                            ? "UPDATE OBSERVATIONS"
                            : "SAVE OBSERVATIONS"}
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </section>

              {/* Private Notes Section */}
              <section className="space-y-4 pt-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-neutral-soft text-medic-dark rounded-xl border border-gray-100">
                    <Clock size={18} />
                  </div>
                  <h2 className="text-xl font-black text-gray-900 tracking-tight">
                    Private Doctor Notes
                  </h2>
                </div>
                <div className="relative">
                  <textarea
                    value={doctorNotes}
                    onChange={(e) => setDoctorNotes(e.target.value)}
                    placeholder="Small notes for yourself (not shared with patient)..."
                    className="w-full h-32 bg-neutral-soft hover:bg-white focus:bg-white border-2 border-transparent focus:border-medic-dark rounded-[2rem] p-8 outline-none transition-all font-medium text-gray-900 text-sm leading-relaxed shadow-inner"
                  />
                  <div className="absolute bottom-6 right-6">
                    <button
                      onClick={handleSaveNotes}
                      disabled={
                        notesLoading || doctorNotes === (patient?.notes || "")
                      }
                      className={`flex items-center gap-2 px-6 py-3 rounded-xl font-black text-xs tracking-wide transition-all shadow-xl active:scale-[0.98] ${
                        doctorNotes === (patient?.notes || "")
                          ? "bg-gray-100 text-gray-400 cursor-not-allowed shadow-none"
                          : "bg-medic-dark text-white hover:bg-medic-primary shadow-medic-dark/20"
                      }`}
                    >
                      SAVE NOTES
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

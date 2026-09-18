import React, { useEffect, useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FileText,
  Calendar,
  CheckCircle2,
  ChevronLeft,
  Download,
  Activity,
  AlertCircle,
  Heart,
  MessageSquare,
  ChevronDown,
  ChevronUp,
  ShieldAlert,
  Loader2,
  Search,
} from "lucide-react";
import { appointmentApi } from "../services/api";
import { useDispatch, useSelector } from "react-redux";
import { useParams, useNavigate, Link } from "react-router-dom";
import {
  fetchReportDetail,
  fetchReportResult,
} from "../store/slices/reportsSlice";
import {
  fetchRecommendedDoctors,
  bookAppointment,
  verifyPayment,
  initiateKhaltiPayment,
  fetchAvailability,
  fetchAppointments,
} from "../store/slices/appointmentSlice";
import { toast } from "react-toastify";
import { BASE_URL } from "../services/api";

const ViewReportResult = () => {
  const { id } = useParams();
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { currentReport, currentResult, loading, error } = useSelector(
    (state) => state.reports,
  );
  const {
    recommendedDoctors,
    recommendedLoading,
    bookingLoading,
    availability,
    availabilityLoading,
    appointments,
  } = useSelector((state) => state.appointment);
  const { user } = useSelector((state) => state.auth);
  const [openAccordion, setOpenAccordion] = useState("measurements");
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [isBookingModalOpen, setIsBookingModalOpen] = useState(false);
  const [appointmentNote, setAppointmentNote] = useState("");

  useEffect(() => {
    if (id) {
      dispatch(fetchReportDetail(id));
      dispatch(fetchReportResult(id));
      dispatch(fetchAppointments());
    }
  }, [dispatch, id]);

  const hasActiveAppointment = useMemo(() => {
    return appointments.some((appt) =>
      ["PAID", "PENDING"].includes(appt.status),
    );
  }, [appointments]);

  useEffect(() => {
    if (currentResult?.suggested_specialization) {
      dispatch(
        fetchRecommendedDoctors({
          specialization: currentResult.suggested_specialization,
          risk_level: currentResult.risk_level,
        }),
      );
    }
  }, [dispatch, currentResult]);

  const [selectedSlot, setSelectedSlot] = useState(null);
  const [selectedDate, setSelectedDate] = useState("");
  const [bookedSlots, setBookedSlots] = useState([]);

  useEffect(() => {
    if (selectedDoctor) {
      dispatch(fetchAvailability(selectedDoctor.id));
      setSelectedDate("");
      setSelectedSlot(null);
      setBookedSlots([]);
    }
  }, [dispatch, selectedDoctor]);

  // Fetch booked slots when a date is selected
  useEffect(() => {
    if (selectedDoctor && selectedDate) {
      setSelectedSlot(null);
      appointmentApi
        .getBookedSlots(selectedDoctor.id, selectedDate)
        .then((res) => setBookedSlots(Array.isArray(res.data) ? res.data : []))
        .catch(() => setBookedSlots([]));
    }
  }, [selectedDoctor, selectedDate]);

  // Filter availability for the selected date's day of week
  const dayOfWeekMap = { 0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5 }; // JS Sunday=0 → backend Monday=0
  const selectedDayOfWeek = selectedDate
    ? dayOfWeekMap[new Date(selectedDate + "T00:00:00").getDay()]
    : null;

  const filteredSlots = useMemo(() => {
    if (!selectedDate || availability.length === 0) return [];

    // 1. Check for specific date overrides
    const dateSpecificSlots = availability.filter(
      (s) => s.date === selectedDate,
    );
    if (dateSpecificSlots.length > 0) {
      return dateSpecificSlots;
    }

    // 2. Fallback to day_of_week
    return availability.filter(
      (s) => Number(s.day_of_week) === selectedDayOfWeek && !s.date,
    );
  }, [selectedDate, availability, selectedDayOfWeek]);
  const isSlotBooked = (slot) =>
    bookedSlots.some(
      (b) =>
        b.start_time.slice(0, 5) === slot.start_time.slice(0, 5) &&
        b.end_time.slice(0, 5) === slot.end_time.slice(0, 5),
    );
  const todayStr = new Date().toISOString().split("T")[0];

  useEffect(() => {
    if (error) {
      toast.error(error);
    }
  }, [error]);

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  };

  if (loading && !currentResult) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-4">
        <Loader2 className="w-12 h-12 text-medic-dark animate-spin" />
        <p className="text-gray-500 font-medium animate-pulse">
          Consulting our AI medical expert...
        </p>
      </div>
    );
  }

  if (!currentResult && !loading) {
    return (
      <div className="max-w-xl mx-auto py-20 text-center space-y-6">
        <div className="w-20 h-20 bg-red-50 text-red-500 rounded-full flex items-center justify-center mx-auto">
          <AlertCircle className="w-10 h-10" />
        </div>
        <h1 className="text-2xl font-bold text-gray-900">Analysis Not Found</h1>
        <p className="text-gray-500">
          We couldn't find the AI analysis for this report. It might still be
          processing or there was an error.
        </p>
        <Link
          to="/dashboard"
          className="inline-flex items-center gap-2 text-medic-dark font-bold hover:underline"
        >
          <ChevronLeft className="w-4 h-4" /> Back to Dashboard
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-6 py-10 space-y-10">
      {/* Header Section */}
      <header className="space-y-6">
        <nav className="flex items-center gap-2 text-[10px] sm:text-sm font-bold text-gray-400 uppercase tracking-widest">
          <Link
            to="/dashboard"
            className="hover:text-medic-dark transition-colors"
          >
            Dashboard
          </Link>
          <span>/</span>
          <span className="text-medic-dark">View Result</span>
        </nav>

        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-3">
              <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold text-gray-900 leading-tight">
                Report Analysis Result
              </h1>
              <span className="bg-medic-light/50 text-medic-dark px-3 py-1 rounded-full text-xs font-bold border border-medic-dark/10 flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5" /> Completed
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-gray-500 font-medium text-sm sm:text-base">
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4" />
                <span className="truncate max-w-[200px]">
                  {currentReport?.file?.split("/").pop() || "Loading..."}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Calendar className="w-4 h-4" />
                <span>{formatDate(currentReport?.uploaded_at)}</span>
              </div>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row items-center gap-3 w-full md:w-auto">
            <a
              href={`${BASE_URL}${currentReport?.file}`}
              download
              target="_blank"
              rel="noopener noreferrer"
              className="w-full sm:w-auto px-6 py-3 bg-white border-2 border-medic-dark text-medic-dark rounded-xl font-bold hover:bg-medic-light/10 transition-all flex items-center justify-center gap-2"
            >
              <Download className="w-5 h-5" /> Download PDF
            </a>
            <button
              onClick={() => navigate("/dashboard")}
              className="w-full sm:w-auto px-6 py-3 bg-medic-dark text-white rounded-xl font-bold hover:bg-medic-primary transition-all shadow-lg shadow-medic-dark/20"
            >
              Back to Dashboard
            </button>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
        {/* Left Column: Analysis Results */}
        <div className="lg:col-span-2 space-y-8">
          {/* Summary Card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-[2rem] p-8 border border-gray-100 shadow-sm space-y-6"
          >
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold text-gray-900 flex items-center gap-3">
                <Activity className="w-6 h-6 text-medic-dark" /> AI-Generated
                Summary
              </h2>
              <div className="text-right">
                <span className="text-xs font-bold text-gray-400 uppercase tracking-widest block mb-1">
                  Confidence Score
                </span>
                <div className="flex items-center gap-3">
                  <div className="w-32 h-2 bg-neutral-soft rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${currentResult?.confidence_score}%` }}
                      className="h-full bg-medic-dark"
                    />
                  </div>
                  <span className="font-bold text-medic-dark">
                    {currentResult?.confidence_score}%
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-medic-light/10 p-6 rounded-2xl border border-medic-dark/10">
              <h4 className="text-[10px] font-black uppercase tracking-widest text-medic-dark/60 mb-2">
                {user?.role === "DOCTOR"
                  ? "Clinical Insights for Physician"
                  : "AI Medical Summary"}
              </h4>
              <p className="text-gray-800 leading-relaxed text-lg italic font-medium">
                "{currentResult?.summary}"
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {currentResult?.key_findings?.map((finding, i) => (
                <div
                  key={i}
                  className="flex items-start gap-3 p-4 bg-medic-light/20 rounded-2xl border border-medic-light/30"
                >
                  <CheckCircle2 className="w-5 h-5 text-medic-dark shrink-0 mt-0.5" />
                  <span className="text-sm font-medium text-gray-700">
                    {finding}
                  </span>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Doctor's Comment Section */}
          {currentReport?.doctor_comment && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-medic-light/10 rounded-[2rem] p-8 border border-medic-dark/10 shadow-sm space-y-4 relative overflow-hidden"
            >
              <div className="absolute top-0 right-0 p-8 opacity-5">
                <MessageSquare className="w-32 h-32 text-medic-dark" />
              </div>
              <div className="relative z-10">
                <h2 className="text-xl font-black text-gray-900 flex items-center gap-3 mb-4">
                  <MessageSquare className="w-6 h-6 text-medic-primary" />
                  Doctor's Note
                </h2>
                <div className="bg-white/80 backdrop-blur-sm p-6 rounded-2xl border border-medic-dark/5">
                  <p className="text-gray-800 text-lg leading-relaxed font-medium italic">
                    "{currentReport.doctor_comment.comment}"
                  </p>
                  <div className="mt-4 flex items-center justify-end gap-2 text-xs font-bold text-medic-dark uppercase tracking-wider">
                    <span>
                      - Dr.{" "}
                      {currentReport.doctor_comment.doctor_name || "Doctor"}
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Expandable Sections */}
          <div className="space-y-4">
            {/* 1. Detected Conditions */}
            <div className="bg-white rounded-3xl border border-gray-100 shadow-sm overflow-hidden">
              <button
                onClick={() =>
                  setOpenAccordion(
                    openAccordion === "conditions" ? null : "conditions",
                  )
                }
                className="w-full px-8 py-6 flex items-center justify-between hover:bg-neutral-soft/30 transition-colors"
              >
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 bg-blue-50 text-blue-500 rounded-xl flex items-center justify-center">
                    <Search className="w-5 h-5" />
                  </div>
                  <span className="text-lg font-bold text-gray-900">
                    Detected Conditions
                  </span>
                </div>
                {openAccordion === "conditions" ? (
                  <ChevronUp className="w-5 h-5 text-gray-400" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-gray-400" />
                )}
              </button>
              <AnimatePresence>
                {openAccordion === "conditions" && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="px-8 pb-8 space-y-4"
                  >
                    {currentResult?.conditions?.map((condition, i) => (
                      <div
                        key={i}
                        className="p-5 bg-blue-50/30 rounded-2xl border border-blue-100 space-y-1"
                      >
                        <h4 className="font-bold text-blue-900">
                          {condition.name}
                        </h4>
                        <p className="text-sm text-blue-700/80 leading-relaxed">
                          {condition.details}
                        </p>
                      </div>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* 2. Extracted Data (OCR) - Doctor Only */}
            {user?.role === "DOCTOR" && currentReport?.extracted_data && (
              <div className="bg-white rounded-3xl border border-gray-100 shadow-sm overflow-hidden">
                <button
                  onClick={() =>
                    setOpenAccordion(openAccordion === "ocr" ? null : "ocr")
                  }
                  className="w-full px-8 py-6 flex items-center justify-between hover:bg-neutral-soft/30 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 bg-green-50 text-green-500 rounded-xl flex items-center justify-center">
                      <FileText className="w-5 h-5" />
                    </div>
                    <span className="text-lg font-bold text-gray-900">
                      Extracted OCR Values
                    </span>
                  </div>
                  {openAccordion === "ocr" ? (
                    <ChevronUp className="w-5 h-5 text-gray-400" />
                  ) : (
                    <ChevronDown className="w-5 h-5 text-gray-400" />
                  )}
                </button>
                <AnimatePresence>
                  {openAccordion === "ocr" && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="px-8 pb-8"
                    >
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                        {Object.entries(
                          currentReport.extracted_data.final_data || {},
                        ).map(([key, val], i) => (
                          <div
                            key={i}
                            className="p-4 bg-neutral-soft rounded-2xl border border-gray-100"
                          >
                            <span className="block text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1">
                              {key}
                            </span>
                            <span className="font-bold text-gray-900">
                              {String(val)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Preview & AI Chat */}
        <div className="space-y-8">
          {/* AI Chat Panel */}
          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-gradient-to-tr from-[#1F7A5B] via-[#00A86B] to-[#4ADE80] rounded-[2rem] p-8 text-white space-y-6 relative overflow-hidden shadow-[0_0_40px_rgba(0,168,107,0.3)] border border-white/20"
          >
            <div className="relative z-10 flex flex-col gap-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-white/20 rounded-2xl flex items-center justify-center backdrop-blur-md">
                  <MessageSquare className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="font-bold text-lg leading-none mb-1">
                    Ask MediScan AI
                  </h3>
                  <div className="flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 rounded-full bg-[#4ADE80] animate-pulse shadow-[0_0_8px_#4ADE80]" />
                    <span className="text-[10px] text-white/90 font-bold uppercase tracking-widest">
                      Health Expert Online
                    </span>
                  </div>
                </div>
              </div>

              <p className="text-sm text-white/95 leading-relaxed font-semibold italic">
                "I've just scanned your medical report. Ready to uncover what
                these numbers mean for your health in plain English?"
              </p>

              <button
                onClick={() => {
                  const chatbotBtn = document.querySelector(
                    "button.fixed.bottom-8.right-8",
                  );
                  if (chatbotBtn) chatbotBtn.click();
                }}
                className="w-full py-4 bg-white text-[#1F7A5B] rounded-2xl font-extrabold hover:bg-medic-light transition-all active:scale-95 shadow-xl shadow-black/10 flex items-center justify-center gap-2"
              >
                Let's Simplify My Report
              </button>

              <div className="p-4 bg-white/5 rounded-2xl border border-white/10 space-y-2">
                <div className="text-sm text-white/95 leading-relaxed font-semibold italic">
                  <AlertCircle className="w-3.5 h-3.5" /> Medical Disclaimer
                </div>
                <p className="text-sm text-white/95 leading-relaxed font-semibold italic">
                  This analysis is AI-generated and not a medical diagnosis.
                  Consult a certified doctor for medical advice regarding your
                  healthcare.
                </p>
              </div>
            </div>

            {/* Decorative Background Icon */}
            <Heart className="absolute -right-10 -bottom-10 w-40 h-40 text-white/5 rotate-12" />
          </motion.div>

          {/* Recommended Doctors Section */}
          {currentResult && (
            <div className="bg-white rounded-[2rem] p-8 border border-gray-100 shadow-sm space-y-6">
              <h3 className="text-xl font-bold text-gray-900 flex items-center gap-3">
                <Activity className="w-6 h-6 text-medic-dark" /> Recommended
                Specialists
              </h3>

              <div className="space-y-4">
                {hasActiveAppointment && (
                  <div className="p-4 bg-amber-50 border border-amber-200 rounded-2xl flex items-start gap-3">
                    <AlertCircle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
                    <p className="text-sm text-amber-800 font-medium">
                      You already have an active appointment or payment pending.
                      Please visit your{" "}
                      <Link to="/appointments" className="underline font-bold">
                        appointments
                      </Link>{" "}
                      to manage it.
                    </p>
                  </div>
                )}

                {recommendedLoading ? (
                  <div className="flex justify-center py-10">
                    <Loader2 className="w-8 h-8 text-medic-dark animate-spin" />
                  </div>
                ) : recommendedDoctors.length > 0 ? (
                  recommendedDoctors.map((doc) => (
                    <div
                      key={doc.id}
                      className="p-4 bg-neutral-soft/30 rounded-2xl border border-gray-100 space-y-3"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-12 h-12 bg-medic-dark rounded-xl flex items-center justify-center text-white font-bold">
                          {doc.first_name[0]}
                          {doc.last_name[0]}
                        </div>
                        <div>
                          <h4 className="font-bold text-gray-900">
                            Dr. {doc.first_name} {doc.last_name}
                          </h4>
                          <p className="text-xs text-medic-dark font-bold uppercase">
                            {doc.specialization?.replace("_", " ")}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="font-bold text-gray-500">
                          Fee: Rs. {doc.consultation_fee}
                        </span>
                        <button
                          onClick={() => {
                            if (hasActiveAppointment) return;
                            setSelectedDoctor(doc);
                            setIsBookingModalOpen(true);
                          }}
                          disabled={hasActiveAppointment}
                          className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${
                            hasActiveAppointment
                              ? "bg-gray-100 text-gray-400 cursor-not-allowed border border-gray-200"
                              : "bg-medic-dark text-white hover:bg-medic-primary shadow-sm"
                          }`}
                        >
                          {hasActiveAppointment
                            ? "Booking Restricted"
                            : "Book Now"}
                        </button>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-gray-500 italic text-center py-4">
                    No specialists found for this condition.
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Booking Modal */}
      <AnimatePresence>
        {isBookingModalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center px-4 md:px-6">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsBookingModalOpen(false)}
              className="absolute inset-0 bg-black/40 backdrop-blur-sm"
            />
            <motion.div
              initial={{ scale: 0.95, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.95, opacity: 0, y: 20 }}
              className="relative w-full max-w-lg bg-white rounded-[2.5rem] shadow-2xl overflow-hidden"
            >
              <div className="p-8 md:p-10 space-y-6">
                <div className="text-center space-y-2">
                  <h2 className="text-2xl font-bold text-gray-900">
                    Book Appointment
                  </h2>
                  <p className="text-gray-500">
                    Consultation with Dr. {selectedDoctor?.first_name}{" "}
                    {selectedDoctor?.last_name}
                  </p>
                </div>

                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-neutral-soft rounded-2xl border border-gray-100">
                      <span className="block text-gray-400 font-bold uppercase text-[10px] mb-1">
                        Fee
                      </span>
                      <span className="text-lg font-bold text-medic-dark">
                        Rs. {selectedDoctor?.consultation_fee}
                      </span>
                    </div>
                    <div className="p-4 bg-neutral-soft rounded-2xl border border-gray-100">
                      <span className="block text-gray-400 font-bold uppercase text-[10px] mb-1">
                        Specialty
                      </span>
                      <span className="text-sm font-bold text-gray-900">
                        {selectedDoctor?.specialization?.replace("_", " ")}
                      </span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs font-bold text-gray-400 uppercase tracking-widest px-2">
                      Select Date
                    </label>
                    <input
                      type="date"
                      value={selectedDate}
                      min={todayStr}
                      onChange={(e) => setSelectedDate(e.target.value)}
                      className="w-full p-3 bg-neutral-soft rounded-2xl border border-gray-100 text-sm font-bold text-medic-dark outline-none focus:border-medic-dark/30 transition-all"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs font-bold text-gray-400 uppercase tracking-widest px-2 text-center block">
                      Available Slots
                      {selectedDate && selectedDayOfWeek !== null && (
                        <span className="ml-2 text-medic-dark normal-case">
                          (
                          {
                            [
                              "Monday",
                              "Tuesday",
                              "Wednesday",
                              "Thursday",
                              "Friday",
                              "Saturday",
                              "Sunday",
                            ][selectedDayOfWeek]
                          }
                          )
                        </span>
                      )}
                    </label>
                    <div className="flex flex-wrap justify-center gap-2">
                      {!selectedDate ? (
                        <p className="text-xs text-gray-400 italic">
                          Please select a date first.
                        </p>
                      ) : availabilityLoading ? (
                        <Loader2 className="w-5 h-5 animate-spin text-medic-dark" />
                      ) : filteredSlots.length > 0 ? (
                        filteredSlots.map((slot, i) => {
                          const booked = isSlotBooked(slot);
                          return (
                            <button
                              key={i}
                              disabled={booked}
                              onClick={() => !booked && setSelectedSlot(slot)}
                              className={`px-4 py-2 rounded-xl text-xs font-bold transition-all border ${
                                booked
                                  ? "bg-red-50 text-red-400 border-red-200 cursor-not-allowed line-through"
                                  : selectedSlot === slot
                                    ? "bg-medic-dark text-white border-medic-dark shadow-md scale-105"
                                    : "bg-neutral-soft text-gray-600 border-gray-100 hover:border-medic-dark/20"
                              }`}
                              title={booked ? "Already booked" : ""}
                            >
                              {slot.start_time.slice(0, 5)} -{" "}
                              {slot.end_time.slice(0, 5)}
                              {booked && " ✕"}
                            </button>
                          );
                        })
                      ) : (
                        <p className="text-xs text-gray-400 italic">
                          No slots available on this day.
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs font-bold text-gray-400 uppercase tracking-widest px-2">
                      Reason (Optional)
                    </label>
                    <textarea
                      value={appointmentNote}
                      onChange={(e) => setAppointmentNote(e.target.value)}
                      placeholder="briefly describe your symptoms or reason for visit..."
                      className="w-full p-6 bg-neutral-soft border-transparent focus:bg-white focus:border-medic-dark/20 rounded-3xl text-sm outline-none transition-all min-h-[120px]"
                    />
                  </div>
                </div>

                <div className="flex gap-4 pt-4">
                  <button
                    onClick={() => setIsBookingModalOpen(false)}
                    className="flex-1 py-4 bg-gray-100 text-gray-500 rounded-2xl font-bold hover:bg-gray-200 transition-all"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={async () => {
                      if (!selectedDate)
                        return toast.warning("Please select a date");
                      if (!selectedSlot)
                        return toast.warning("Please select a time slot");
                      try {
                        const apptData = {
                          doctor: selectedDoctor.id,
                          appointment_date: selectedDate,
                          start_time: selectedSlot.start_time,
                          end_time: selectedSlot.end_time,
                          notes: appointmentNote,
                        };
                        const appt = await dispatch(
                          bookAppointment(apptData),
                        ).unwrap();

                        // Initialize Khalti Payment (Redirect Flow)
                        try {
                          const paymentData = {
                            appointmentId: appt.id,
                            returnUrl: `${window.location.origin}/payment/success`,
                            websiteUrl: window.location.origin,
                          };

                          const khaltiResponse = await dispatch(
                            initiateKhaltiPayment(paymentData),
                          ).unwrap();

                          if (khaltiResponse.payment_url) {
                            window.location.href = khaltiResponse.payment_url;
                          } else {
                            toast.error(
                              "Failed to get payment URL from Khalti",
                            );
                          }
                        } catch (paymentError) {
                          console.error(
                            "Payment initiation failed:",
                            paymentError,
                          );
                          toast.error(
                            "Failed to initiate payment. Please try again.",
                          );
                        }
                      } catch (error) {
                        console.error("Booking error:", error);
                        // Error toast is handled by slice
                      }
                    }}
                    disabled={bookingLoading}
                    className="flex-1 py-4 bg-medic-dark text-white rounded-2xl font-bold hover:bg-medic-primary transition-all shadow-lg shadow-medic-dark/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {bookingLoading ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      "Confirm & Pay"
                    )}
                  </button>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ViewReportResult;

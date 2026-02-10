import React, { useEffect, useState } from "react";
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
  Clipboard,
  MessageSquare,
  ChevronDown,
  ChevronUp,
  ShieldAlert,
  Loader2,
  Search,
} from "lucide-react";
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
  fetchAvailability,
} from "../store/slices/appointmentSlice";
import { toast } from "react-toastify";

const ViewReportResult = () => {
  const { id } = useParams();
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { currentReport, currentResult, loading, error } = useSelector(
    (state) => state.reports,
  );
  const { recommendedDoctors, recommendedLoading, bookingLoading, availability, loading: availabilityLoading } = useSelector(
    (state) => state.appointment,
  );
  const { user } = useSelector((state) => state.auth);
  const [openAccordion, setOpenAccordion] = useState("measurements");
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [isBookingModalOpen, setIsBookingModalOpen] = useState(false);
  const [appointmentNote, setAppointmentNote] = useState("");

  useEffect(() => {
    if (id) {
      dispatch(fetchReportDetail(id));
      dispatch(fetchReportResult(id));
    }
  }, [dispatch, id]);

  useEffect(() => {
    if (currentResult?.suggested_specialization) {
      dispatch(fetchRecommendedDoctors(currentResult.suggested_specialization));
    }
  }, [dispatch, currentResult]);

  const [selectedSlot, setSelectedSlot] = useState(null);

  useEffect(() => {
    if (selectedDoctor) {
      dispatch(fetchAvailability(selectedDoctor.id));
    }
  }, [dispatch, selectedDoctor]);

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
                  {currentReport?.file?.split("/").pop() || 'Loading...'}
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
              href={`http://localhost:8000${currentReport?.file}`}
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

            <p className="text-gray-600 leading-relaxed text-lg italic">
              "{currentResult?.summary}"
            </p>

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

            {/* 2. Important Measurements */}
            <div className="bg-white rounded-3xl border border-gray-100 shadow-sm overflow-hidden">
              <button
                onClick={() =>
                  setOpenAccordion(
                    openAccordion === "measurements" ? null : "measurements",
                  )
                }
                className="w-full px-8 py-6 flex items-center justify-between hover:bg-neutral-soft/30 transition-colors"
              >
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 bg-medic-light/50 text-medic-dark rounded-xl flex items-center justify-center">
                    <Clipboard className="w-5 h-5" />
                  </div>
                  <span className="text-lg font-bold text-gray-900">
                    Important Measurements
                  </span>
                </div>
                {openAccordion === "measurements" ? (
                  <ChevronUp className="w-5 h-5 text-gray-400" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-gray-400" />
                )}
              </button>
              <AnimatePresence>
                {openAccordion === "measurements" && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="px-8 pb-8"
                  >
                    <div className="overflow-x-auto">
                      <table className="w-full text-left">
                        <thead>
                          <tr className="border-b border-gray-100 text-xs font-bold text-gray-400 uppercase tracking-widest">
                            <th className="py-4">Parameter</th>
                            <th className="py-4">Result</th>
                            <th className="py-4">Ref. Range</th>
                            <th className="py-4 text-right">Status</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-50">
                          {currentReport?.extracted_data?.final_data?.tests?.map(
                            (test, i) => (
                              <tr key={i}>
                                <td className="py-4 font-bold text-gray-900">
                                  {test.name}
                                </td>
                                <td className="py-4 text-medic-dark font-bold">
                                  {test.value}{" "}
                                  <span className="text-xs text-gray-400">
                                    {test.unit}
                                  </span>
                                </td>
                                <td className="py-4 text-sm text-gray-500">
                                  {test.reference_range || "--"}
                                </td>
                                <td className="py-4 text-right">
                                  <span
                                    className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${test.status === "Normal"
                                      ? "bg-green-100 text-green-700"
                                      : "bg-orange-100 text-orange-700"
                                      }`}
                                  >
                                    {test.status}
                                  </span>
                                </td>
                              </tr>
                            ),
                          )}
                        </tbody>
                      </table>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* 3. Risk Indicators */}
            <div className="bg-white rounded-3xl border border-gray-100 shadow-sm overflow-hidden">
              <button
                onClick={() =>
                  setOpenAccordion(openAccordion === "risk" ? null : "risk")
                }
                className="w-full px-8 py-6 flex items-center justify-between hover:bg-neutral-soft/30 transition-colors"
              >
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 bg-red-50 text-red-500 rounded-xl flex items-center justify-center">
                    <ShieldAlert className="w-5 h-5" />
                  </div>
                  <span className="text-lg font-bold text-gray-900">
                    Risk Indicators
                  </span>
                </div>
                {openAccordion === "risk" ? (
                  <ChevronUp className="w-5 h-5 text-gray-400" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-gray-400" />
                )}
              </button>
              <AnimatePresence>
                {openAccordion === "risk" && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="px-8 pb-8 flex flex-col sm:flex-row items-center gap-6"
                  >
                    <div className="space-y-1 w-full sm:w-auto">
                      <span className="text-[10px] sm:text-sm font-bold text-gray-400 uppercase tracking-widest block text-center sm:text-left">
                        Identified Risk Level
                      </span>
                      <div className="flex flex-col sm:flex-row items-center gap-4">
                        <span
                          className={`w-full sm:w-auto px-6 py-2 rounded-2xl text-lg font-bold text-center ${currentResult?.risk_level === "Low"
                            ? "bg-green-100 text-green-700"
                            : currentResult?.risk_level === "Medium"
                              ? "bg-orange-100 text-orange-700"
                              : "bg-red-100 text-red-700"
                            }`}
                        >
                          {currentResult?.risk_level} Risk
                        </span>
                        <p className="text-sm text-gray-500 max-w-xs italic text-center sm:text-left">
                          "Based on available laboratory markers, your current
                          risk assessment is categorized as{" "}
                          {currentResult?.risk_level}."
                        </p>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>

        {/* Right Column: Preview & AI Chat */}
        <div className="space-y-8">
          {/* AI Chat Panel */}
          <div className="bg-medic-dark rounded-[2rem] p-8 text-white space-y-6 relative overflow-hidden">
            <div className="relative z-10 flex flex-col gap-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-medic-light/20 rounded-2xl flex items-center justify-center">
                  <MessageSquare className="w-6 h-6 text-medic-accent" />
                </div>
                <div>
                  <h3 className="font-bold text-lg leading-none">
                    AI Health Expert
                  </h3>
                  <span className="text-xs text-medic-accent/70 font-bold uppercase tracking-widest">
                    Active Now
                  </span>
                </div>
              </div>

              <p className="text-sm text-medic-light/80 leading-relaxed font-medium">
                "I've analyzed your results. Would you like me to explain these
                measurements in simpler terms?"
              </p>

              <button className="w-full py-4 bg-medic-accent text-medic-dark rounded-2xl font-bold hover:bg-medic-accent/90 transition-all active:scale-95 shadow-lg shadow-black/20">
                Explain in Simple Terms
              </button>

              <div className="p-4 bg-white/5 rounded-2xl border border-white/10 space-y-2">
                <div className="flex items-center gap-2 text-xs font-bold text-medic-accent uppercase tracking-widest">
                  <AlertCircle className="w-3.5 h-3.5" /> Medical Disclaimer
                </div>
                <p className="text-[10px] text-medic-light/40 leading-normal">
                  This analysis is AI-generated and not a medical diagnosis.
                  Consult a certified doctor for medical advice regarding your
                  healthcare.
                </p>
              </div>
            </div>

            {/* Decorative Background Icon */}
            <Heart className="absolute -right-10 -bottom-10 w-40 h-40 text-white/5 rotate-12" />
          </div>

          {/* Recommended Doctors Section */}
          <div className="bg-white rounded-[2rem] p-8 border border-gray-100 shadow-sm space-y-6">
            <h3 className="text-xl font-bold text-gray-900 flex items-center gap-3">
              <Activity className="w-6 h-6 text-medic-dark" /> Recommended Specialists
            </h3>

            <div className="space-y-4">
              {recommendedLoading ? (
                <div className="flex justify-center py-10">
                  <Loader2 className="w-8 h-8 text-medic-dark animate-spin" />
                </div>
              ) : recommendedDoctors.length > 0 ? (
                recommendedDoctors.map((doc) => (
                  <div key={doc.id} className="p-4 bg-neutral-soft/30 rounded-2xl border border-gray-100 space-y-3">
                    <div className="flex items-center gap-3">
                      <div className="w-12 h-12 bg-medic-dark rounded-xl flex items-center justify-center text-white font-bold">
                        {doc.first_name[0]}{doc.last_name[0]}
                      </div>
                      <div>
                        <h4 className="font-bold text-gray-900">Dr. {doc.first_name} {doc.last_name}</h4>
                        <p className="text-xs text-medic-dark font-bold uppercase">{doc.specialization?.replace('_', ' ')}</p>
                      </div>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-bold text-gray-500">Fee: Rs. {doc.consultation_fee}</span>
                      <button
                        onClick={() => {
                          setSelectedDoctor(doc);
                          setIsBookingModalOpen(true);
                        }}
                        className="px-4 py-2 bg-medic-dark text-white rounded-lg text-xs font-bold hover:bg-medic-primary transition-all"
                      >
                        Book Now
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
                  <h2 className="text-2xl font-bold text-gray-900">Book Appointment</h2>
                  <p className="text-gray-500">Consultation with Dr. {selectedDoctor?.first_name} {selectedDoctor?.last_name}</p>
                </div>

                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="p-4 bg-neutral-soft rounded-2xl border border-gray-100">
                      <span className="block text-gray-400 font-bold uppercase text-[10px] mb-1">Fee</span>
                      <span className="text-lg font-bold text-medic-dark">Rs. {selectedDoctor?.consultation_fee}</span>
                    </div>
                    <div className="p-4 bg-neutral-soft rounded-2xl border border-gray-100">
                      <span className="block text-gray-400 font-bold uppercase text-[10px] mb-1">Specialty</span>
                      <span className="text-sm font-bold text-gray-900">{selectedDoctor?.specialization?.replace('_', ' ')}</span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs font-bold text-gray-400 uppercase tracking-widest px-2 text-center block">Available Slots</label>
                    <div className="flex flex-wrap justify-center gap-2">
                      {availabilityLoading ? (
                        <Loader2 className="w-5 h-5 animate-spin text-medic-dark" />
                      ) : availability.length > 0 ? (
                        availability.map((slot, i) => (
                          <button
                            key={i}
                            onClick={() => setSelectedSlot(slot)}
                            className={`px-4 py-2 rounded-xl text-xs font-bold transition-all border ${selectedSlot === slot ? 'bg-medic-dark text-white border-medic-dark shadow-md scale-105' : 'bg-neutral-soft text-gray-600 border-gray-100 hover:border-medic-dark/20'}`}
                          >
                            {slot.start_time.slice(0, 5)} - {slot.end_time.slice(0, 5)}
                          </button>
                        ))
                      ) : (
                        <p className="text-xs text-gray-400 italic">No slots available for this week.</p>
                      )}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs font-bold text-gray-400 uppercase tracking-widest px-2">Reason (Optional)</label>
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
                      if (!selectedSlot) return toast.warning("Please select a time slot");
                      try {
                        const apptData = {
                          doctor: selectedDoctor.id,
                          appointment_date: new Date().toISOString().split('T')[0], // Placeholder: Today
                          start_time: selectedSlot.start_time,
                          end_time: selectedSlot.end_time,
                          notes: appointmentNote
                        };
                        const appt = await dispatch(bookAppointment(apptData)).unwrap();

                        // Initialize Khalti
                        const config = {
                          publicKey: import.meta.env.VITE_KHALTI_PUBLIC_KEY || "test_public_key_xxxx",
                          productIdentity: appt.id.toString(),
                          productName: `Consultation with Dr. ${selectedDoctor.first_name}`,
                          productUrl: window.location.href,
                          eventHandler: {
                            onSuccess(payload) {
                              dispatch(verifyPayment({
                                appointmentId: appt.id,
                                data: { token: payload.token, amount: payload.amount }
                              })).then(() => {
                                setIsBookingModalOpen(false);
                                navigate('/dashboard');
                              });
                            },
                            onError(error) {
                              console.error(error);
                              toast.error("Khalti payment failed");
                            },
                            onClose() {
                              console.log('widget is closing');
                            }
                          },
                          paymentPreference: ["KHALTI", "EBANKING", "MOBILE_BANKING", "CONNECT_IPS", "SCT"],
                        };
                        const checkout = new window.KhaltiCheckout(config);
                        checkout.show({ amount: parseInt(selectedDoctor.consultation_fee) * 100 });
                      } catch (err) {
                        toast.error(err?.message || "Booking failed");
                      }
                    }}
                    disabled={bookingLoading}
                    className="flex-1 py-4 bg-medic-dark text-white rounded-2xl font-bold hover:bg-medic-primary transition-all flex items-center justify-center gap-2"
                  >
                    {bookingLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : "Pay & Book"}
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

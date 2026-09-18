import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  FileText,
  Activity,
  MessageSquare,
  Send,
  Loader2,
  AlertCircle,
  Clock,
  ChevronDown,
  ChevronUp,
  Download,
} from "lucide-react";
import { useDispatch, useSelector } from "react-redux";
import {
  fetchPatientReports,
  submitDoctorComment,
} from "../../store/slices/doctorSlice";
import { BASE_URL } from "../../services/api";

const PatientReportSummary = ({ patientId, onClose, patientName }) => {
  const dispatch = useDispatch();
  const { currentPatientReports, reportsLoading, commentLoading, error } =
    useSelector((state) => state.doctor);
  const [expandedReport, setExpandedReport] = useState(null);
  const [commentText, setCommentText] = useState("");
  const [activeReportId, setActiveReportId] = useState(null);

  useEffect(() => {
    if (patientId) {
      dispatch(fetchPatientReports(patientId));
    }
  }, [dispatch, patientId]);

  // Auto-expand the first report if available
  useEffect(() => {
    if (currentPatientReports.length > 0 && !expandedReport) {
      setExpandedReport(currentPatientReports[0].id);
    }
  }, [currentPatientReports]);

  const handleToggleExpand = (id) => {
    setExpandedReport(expandedReport === id ? null : id);
  };

  const handleAddComment = (reportId, existingComment) => {
    setActiveReportId(reportId);
    setCommentText(existingComment || "");
  };

  const handleSubmitComment = async (reportId) => {
    if (!commentText.trim()) return;

    try {
      await dispatch(
        submitDoctorComment({
          report: reportId,
          comment: commentText,
        }),
      ).unwrap();
      setActiveReportId(null);
      setCommentText("");
    } catch (error) {
      console.error("Failed to add comment:", error);
    }
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0, y: 20 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.9, opacity: 0, y: 20 }}
          className="bg-white rounded-3xl w-full max-w-2xl max-h-[85vh] overflow-hidden shadow-2xl flex flex-col"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="p-6 border-b border-gray-100 flex justify-between items-center bg-gray-50/50">
            <div>
              <h2 className="text-xl font-black text-gray-900">
                Medical Summary
              </h2>
              <p className="text-sm text-gray-500 font-medium">
                Patient: <span className="text-medic-dark">{patientName}</span>
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-full hover:bg-gray-100 text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {reportsLoading ? (
              <div className="flex flex-col items-center justify-center py-20 text-gray-400">
                <Loader2 className="w-10 h-10 animate-spin mb-3 text-medic-dark" />
                <p>Loading patient records...</p>
              </div>
            ) : error ? (
              <div className="text-center py-20 px-8 bg-red-50 rounded-3xl border border-dashed border-red-200">
                <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
                <h3 className="text-gray-900 font-bold mb-1">
                  Failed to Load Reports
                </h3>
                <p className="text-gray-500 text-sm">
                  {typeof error === "string"
                    ? error
                    : "Could not fetch patient reports. Please try logging out and back in."}
                </p>
              </div>
            ) : currentPatientReports.length > 0 ? (
              currentPatientReports.map((report) => {
                const hasResult = report.result || report.ai_analysis;
                const resultData = report.result || report.ai_analysis;
                const isExpanded = expandedReport === report.id;

                return (
                  <div
                    key={report.id}
                    className={`border rounded-2xl transition-all ${
                      isExpanded
                        ? "border-medic-dark/20 bg-medic-light/5 shadow-md"
                        : "border-gray-100 bg-white hover:border-gray-200"
                    }`}
                  >
                    <div
                      className="p-4 flex items-center justify-between cursor-pointer"
                      onClick={() => handleToggleExpand(report.id)}
                    >
                      <div className="flex items-center gap-4">
                        <div
                          className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                            hasResult
                              ? "bg-green-100 text-green-600"
                              : "bg-gray-100 text-gray-400"
                          }`}
                        >
                          {hasResult ? (
                            <Activity className="w-5 h-5" />
                          ) : (
                            <FileText className="w-5 h-5" />
                          )}
                        </div>
                        <div>
                          <h4 className="font-bold text-gray-900 text-sm">
                            {report.file?.split("/").pop() || "Unnamed Report"}
                          </h4>
                          <div className="flex items-center gap-2 text-xs text-gray-500">
                            <Clock className="w-3 h-3" />
                            {new Date(report.uploaded_at).toLocaleDateString()}
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <a
                            href={`${BASE_URL}${report.file}`}
                            download
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            className="p-1.5 text-gray-400 hover:text-medic-dark hover:bg-medic-light/20 rounded-lg transition-all"
                            title="Download PDF"
                          >
                            <Download className="w-4 h-4" />
                          </a>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {report.doctor_comment && (
                          <div className="px-2 py-1 bg-medic-dark text-white text-[10px] font-bold rounded-full flex items-center gap-1">
                            <MessageSquare className="w-3 h-3" /> Commented
                          </div>
                        )}
                        {isExpanded ? (
                          <ChevronUp className="w-5 h-5 text-gray-400" />
                        ) : (
                          <ChevronDown className="w-5 h-5 text-gray-400" />
                        )}
                      </div>
                    </div>

                    <AnimatePresence>
                      {isExpanded && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: "auto", opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          className="overflow-hidden"
                        >
                          <div className="p-4 pt-0 border-t border-dashed border-gray-100 mt-2">
                            {hasResult ? (
                              <div className="space-y-4 pt-4">
                                {/* AI Summary Section */}
                                <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
                                  <h5 className="flex items-center gap-2 text-xs font-black uppercase text-medic-dark mb-3 tracking-wider">
                                    <Activity className="w-4 h-4" /> AI Analysis
                                  </h5>
                                  <p className="text-sm text-gray-600 leading-relaxed">
                                    {resultData.summary}
                                  </p>
                                  <div className="mt-3 flex flex-wrap gap-2">
                                    <span className="px-3 py-1 bg-gray-100 text-gray-600 rounded-lg text-xs font-bold">
                                      Risk: {resultData?.risk_level || "N/A"}
                                    </span>
                                    {resultData?.suggested_specialization && (
                                      <span className="px-3 py-1 bg-blue-50 text-blue-600 rounded-lg text-xs font-bold">
                                        {Array.isArray(
                                          resultData.suggested_specialization,
                                        )
                                          ? resultData.suggested_specialization.join(
                                              ", ",
                                            )
                                          : resultData.suggested_specialization.replace(
                                              /_/g,
                                              " ",
                                            )}
                                      </span>
                                    )}
                                  </div>
                                </div>

                                {/* Doctor Comment Section */}
                                <div className="bg-medic-light/10 rounded-xl p-4 border border-medic-dark/5">
                                  <div className="flex items-center justify-between mb-3">
                                    <h5 className="flex items-center gap-2 text-xs font-black uppercase text-gray-900 tracking-wider">
                                      <MessageSquare className="w-4 h-4 text-medic-primary" />
                                      Doctor's Note
                                    </h5>
                                    {!activeReportId && (
                                      <button
                                        onClick={() =>
                                          handleAddComment(
                                            report.id,
                                            report.doctor_comment?.comment,
                                          )
                                        }
                                        className="text-xs font-bold text-medic-dark hover:underline"
                                      >
                                        {report.doctor_comment
                                          ? "Edit Note"
                                          : "Add Note"}
                                      </button>
                                    )}
                                  </div>

                                  {activeReportId === report.id ? (
                                    <div className="space-y-3">
                                      <textarea
                                        value={commentText}
                                        onChange={(e) =>
                                          setCommentText(e.target.value)
                                        }
                                        placeholder="Add your clinical observations for the patient..."
                                        className="w-full p-3 bg-white border border-gray-200 rounded-xl text-sm focus:ring-2 focus:ring-medic-dark/20 focus:border-medic-dark outline-none transition-all placeholder:text-gray-400 min-h-[100px]"
                                        autoFocus
                                      />
                                      <div className="flex gap-2 justify-end">
                                        <button
                                          onClick={() =>
                                            setActiveReportId(null)
                                          }
                                          className="px-4 py-2 text-xs font-bold text-gray-500 hover:bg-gray-100 rounded-lg transition-colors"
                                        >
                                          Cancel
                                        </button>
                                        <button
                                          onClick={() =>
                                            handleSubmitComment(report.id)
                                          }
                                          disabled={commentLoading}
                                          className="px-4 py-2 bg-medic-dark text-white text-xs font-bold rounded-lg hover:bg-medic-primary transition-colors flex items-center gap-2 disabled:opacity-50"
                                        >
                                          {commentLoading && (
                                            <Loader2 className="w-3 h-3 animate-spin" />
                                          )}
                                          Save Note
                                        </button>
                                      </div>
                                    </div>
                                  ) : report.doctor_comment ? (
                                    <p className="text-sm text-gray-700 italic border-l-2 border-medic-primary pl-3 py-1">
                                      "{report.doctor_comment.comment}"
                                    </p>
                                  ) : (
                                    <p className="text-sm text-gray-400 italic">
                                      No notes added yet.
                                    </p>
                                  )}
                                </div>
                              </div>
                            ) : (
                              <div className="py-6 text-center">
                                <AlertCircle className="w-8 h-8 text-gray-300 mx-auto mb-2" />
                                <p className="text-sm text-gray-400">
                                  Processing report analysis...
                                </p>
                              </div>
                            )}
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                );
              })
            ) : (
              <div className="text-center py-20 px-8 bg-gray-50 rounded-3xl border border-dashed border-gray-200">
                <FileText className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                <h3 className="text-gray-900 font-bold mb-1">
                  No Records Found
                </h3>
                <p className="text-gray-500 text-sm">
                  This patient hasn't uploaded any reports yet.
                </p>
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default PatientReportSummary;

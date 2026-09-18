import { useDispatch, useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";
import {
  Search,
  ShieldCheck,
  Filter,
  Stethoscope,
  Activity,
  ArrowRight,
  Sparkles,
  Lock,
} from "lucide-react";
import { fetchRecommendedDoctors } from "../store/slices/appointmentSlice";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect } from "react";

const FindDoctors = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { recommendedDoctors: doctors, recommendedLoading: loading } =
    useSelector((state) => state.appointment);
  const { isAuthenticated } = useSelector((state) => state.auth);

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedSpecialty, setSelectedSpecialty] = useState("all");

  useEffect(() => {
    dispatch(fetchRecommendedDoctors({}));
  }, [dispatch]);

  const filteredDoctors = doctors.filter((doc) => {
    const matchesSearch =
      `${doc.first_name} ${doc.last_name}`
        .toLowerCase()
        .includes(searchQuery.toLowerCase()) ||
      doc.specialization?.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesSpecialty =
      selectedSpecialty === "all" || doc.specialization === selectedSpecialty;
    return matchesSearch && matchesSpecialty;
  });

  const specialties = [
    "all",
    "CARDIOLOGIST",
    "ENDOCRINOLOGIST",
    "HEPATOLOGIST",
    "NEPHROLOGIST",
    "HEMATOLOGIST",
    "GENERAL_PHYSICIAN",
  ];

  const handleBookClick = (doctorId) => {
    if (!isAuthenticated) {
      navigate("/login", { state: { from: `/book-appointment/${doctorId}` } });
    } else {
      navigate(`/book-appointment/${doctorId}`);
    }
  };

  return (
    <div className="min-h-screen bg-neutral-background">
      {/* Search Header */}
      <div className="bg-medic-dark pt-32 pb-24 px-6 rounded-b-[3rem] shadow-xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-[50vw] h-[50vh] bg-white/5 rounded-full blur-[100px] -translate-y-1/2 translate-x-1/2" />
        <div className="absolute bottom-0 left-0 w-[40vw] h-[40vh] bg-medic-accent/10 rounded-full blur-[80px] translate-y-1/2 -translate-x-1/2" />

        <div className="max-w-7xl mx-auto relative z-10 space-y-8">
          <div className="text-center space-y-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="inline-flex items-center gap-2 px-4 py-1.5 bg-white/10 border border-white/20 rounded-full text-white/90 font-bold text-xs tracking-widest uppercase backdrop-blur-md"
            >
              <Sparkles className="w-3 h-3 text-medic-accent" />
              Verified Specialists
            </motion.div>
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="text-4xl md:text-5xl font-black text-white tracking-tight"
            >
              Find your <span className="text-medic-accent">Expert.</span>
            </motion.h1>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-lg text-medic-light/80 max-w-2xl mx-auto leading-relaxed"
            >
              Connect with top-tier medical professionals vetted for excellence.
            </motion.p>
          </div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-white rounded-[2rem] p-3 shadow-2xl shadow-medic-dark/20 border border-white/20 flex flex-col md:flex-row gap-2 max-w-4xl mx-auto backdrop-blur-xl"
          >
            <div className="flex-1 relative group">
              <Search className="absolute left-5 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 group-focus-within:text-medic-dark transition-colors" />
              <input
                type="text"
                placeholder="Search doctors, specialties..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-14 pr-6 py-4 bg-gray-50 border-transparent focus:bg-white rounded-[1.5rem] text-gray-900 placeholder:text-gray-400 outline-none transition-all font-medium"
              />
            </div>
            <div className="md:w-72 relative">
              <Filter className="absolute left-5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
              <select
                value={selectedSpecialty}
                onChange={(e) => setSelectedSpecialty(e.target.value)}
                className="w-full pl-12 pr-10 py-4 bg-gray-50 border-transparent focus:bg-white rounded-[1.5rem] appearance-none text-gray-700 font-bold text-sm outline-none transition-all cursor-pointer hover:bg-gray-100"
              >
                {specialties.map((spec) => (
                  <option key={spec} value={spec}>
                    {spec === "all"
                      ? "All Specialties"
                      : spec.replace("_", " ")}
                  </option>
                ))}
              </select>
            </div>
          </motion.div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-16">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          <AnimatePresence mode="popLayout">
            {loading ? (
              Array.from({ length: 6 }).map((_, i) => (
                <div
                  key={i}
                  className="bg-white rounded-[2.5rem] p-8 border border-gray-100 animate-pulse space-y-6"
                >
                  <div className="flex gap-4 items-center">
                    <div className="w-16 h-16 bg-gray-100 rounded-2xl" />
                    <div className="space-y-2 flex-1">
                      <div className="h-4 bg-gray-100 rounded w-3/4" />
                      <div className="h-3 bg-gray-100 rounded w-1/2" />
                    </div>
                  </div>
                  <div className="h-24 bg-gray-100 rounded-2xl" />
                  <div className="h-12 bg-gray-100 rounded-xl" />
                </div>
              ))
            ) : filteredDoctors.length > 0 ? (
              filteredDoctors.map((doc, idx) => (
                <motion.div
                  key={doc.id}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ delay: idx * 0.05 }}
                  className="group bg-white rounded-[2.5rem] p-8 border border-gray-100 hover:border-medic-dark/10 hover:shadow-2xl hover:shadow-medic-dark/5 transition-all relative overflow-hidden flex flex-col"
                >
                  <div className="relative z-10 flex flex-col h-full">
                    {/* Header */}
                    <div className="flex items-start justify-between mb-8">
                      <div className="flex gap-5">
                        <div className="w-16 h-16 bg-gradient-to-br from-medic-dark to-medic-primary rounded-2xl flex items-center justify-center text-white text-xl font-black shadow-lg shadow-medic-dark/20 group-hover:scale-110 transition-transform duration-300">
                          {doc.first_name[0]}
                          {doc.last_name[0]}
                        </div>
                        <div>
                          <h3 className="text-xl font-black text-slate-900 group-hover:text-medic-dark transition-colors capitalize leading-tight mb-1">
                            Dr. {doc.first_name} {doc.last_name}
                          </h3>
                          <div className="flex items-center gap-1.5 text-medic-primary font-bold text-[10px] uppercase tracking-widest">
                            <Stethoscope className="w-3 h-3" />
                            {doc.specialization?.replace("_", " ")}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Stats Card */}
                    <div className="bg-gray-50 rounded-2xl p-4 grid grid-cols-2 gap-px mb-6 border border-gray-100">
                      <div className="px-2 text-center border-r border-gray-200">
                        <p className="text-[10px] text-gray-400 font-black uppercase tracking-widest mb-1">
                          Exp
                        </p>
                        <p className="font-black text-slate-900 text-lg">
                          {doc.experience || 0}
                          <span className="text-xs font-bold text-gray-400 ml-0.5">
                            yrs
                          </span>
                        </p>
                      </div>
                      <div className="px-2 text-center">
                        <p className="text-[10px] text-gray-400 font-black uppercase tracking-widest mb-1">
                          Fee
                        </p>
                        <p className="font-black text-medic-dark text-lg">
                          {doc.consultation_fee || "1000"}
                        </p>
                      </div>
                    </div>

                    <div className="mb-8 space-y-3 flex-1">
                      <div className="flex items-start gap-3 text-sm text-gray-600 font-medium">
                        <Activity className="w-4 h-4 text-medic-accent mt-0.5 shrink-0" />
                        <span className="line-clamp-2 leading-relaxed">
                          {doc.bio ||
                            "Specialist dedicated to providing comprehensive care and accurate diagnostics."}
                        </span>
                      </div>
                      <div className="flex items-center gap-3 text-sm text-gray-600 font-medium">
                        <ShieldCheck className="w-4 h-4 text-emerald-500 shrink-0" />
                        <span>Verified License</span>
                      </div>
                    </div>

                    {/* CTA */}
                    <button
                      onClick={() => handleBookClick(doc.id)}
                      className="w-full mt-auto bg-medic-dark text-white py-4 rounded-2xl font-bold text-sm flex items-center justify-center gap-2 group-hover:bg-medic-primary transition-all shadow-xl shadow-medic-dark/10 hover:shadow-medic-dark/20 hover:-translate-y-1"
                    >
                      {!isAuthenticated ? (
                        <>
                          <Lock className="w-4 h-4" /> Login to Book
                        </>
                      ) : (
                        <>
                          Book Appointment
                          <ArrowRight className="w-4 h-4" />
                        </>
                      )}
                    </button>
                  </div>
                </motion.div>
              ))
            ) : (
              <div className="col-span-full py-32 text-center space-y-8">
                <div className="w-24 h-24 bg-gray-50 rounded-full flex items-center justify-center mx-auto text-gray-300 border-4 border-white shadow-xl">
                  <Search className="w-10 h-10" />
                </div>
                <div>
                  <h3 className="text-2xl font-black text-slate-900 mb-2">
                    No Specialists Found
                  </h3>
                  <p className="text-gray-500 max-w-md mx-auto font-medium">
                    We couldn&apos;t find any verified doctors matching your
                    search criteria. Try adjusting your filters.
                  </p>
                </div>
                <button
                  onClick={() => {
                    setSearchQuery("");
                    setSelectedSpecialty("all");
                  }}
                  className="px-8 py-4 bg-medic-dark text-white rounded-2xl font-bold text-sm hover:translate-y-[-2px] transition-transform shadow-lg shadow-medic-dark/20"
                >
                  Clear all filters
                </button>
              </div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};

export default FindDoctors;

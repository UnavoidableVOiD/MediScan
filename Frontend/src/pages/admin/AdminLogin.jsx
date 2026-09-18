import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useDispatch } from "react-redux";
import { ShieldCheck, Lock, Mail, ArrowRight, Loader2 } from "lucide-react";
import { motion } from "framer-motion";
import { setCredentials } from "../../store/slices/authSlice";
import { adminLogin } from "../../store/slices/adminSlice";

const AdminLogin = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const result = await dispatch(adminLogin(formData)).unwrap();
      dispatch(
        setCredentials({
          user: result.user,
          token: null, // Handled by cookie
        }),
      );

      if (result.user.role === "ADMIN" || result.user.is_superuser) {
        navigate("/admin/dashboard");
      }
    } catch (error) {
      // Errors are handled by the adminSlice thunk
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#F8FAFC] relative overflow-hidden px-6">
      {/* Decorative Background Elements */}
      <div className="absolute top-0 right-0 w-1/3 h-1/3 bg-medic-primary/10 blur-[120px] rounded-full -translate-y-1/2 translate-x-1/2"></div>
      <div className="absolute bottom-0 left-0 w-1/2 h-1/2 bg-indigo-500/5 blur-[120px] rounded-full translate-y-1/2 -translate-x-1/4"></div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-xl w-full"
      >
        <div className="bg-white rounded-[3rem] shadow-2xl shadow-slate-200/60 overflow-hidden border border-slate-100 flex flex-col md:flex-row">
          <div className="p-10 md:p-14 w-full">
            <div className="text-left mb-12">
              <div className="w-14 h-14 bg-[#0F172A] rounded-2xl flex items-center justify-center mb-8 shadow-xl shadow-slate-900/20">
                <ShieldCheck className="w-7 h-7 text-medic-primary" />
              </div>
              <h1 className="text-4xl font-black text-slate-900 tracking-tight leading-tight">
                System Infrastructure Access
              </h1>
              <p className="text-slate-500 mt-4 text-lg font-medium leading-relaxed">
                Enter your administrative credentials to authorize secure
                network management.
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-8">
              <div className="space-y-2">
                <label className="text-[11px] font-black text-slate-400 uppercase tracking-[0.2em] ml-2">
                  Operator Identity
                </label>
                <div className="relative group">
                  <Mail className="absolute left-5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-300 group-focus-within:text-medic-primary transition-colors" />
                  <input
                    type="email"
                    name="email"
                    required
                    value={formData.email}
                    onChange={handleChange}
                    placeholder="operator@mediscan.net"
                    className="w-full pl-14 pr-6 py-5 bg-slate-50 border-2 border-transparent focus:border-medic-primary/20 focus:bg-white rounded-[1.5rem] text-sm font-bold transition-all outline-none"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-[11px] font-black text-slate-400 uppercase tracking-[0.2em] ml-2">
                  Authorization Key
                </label>
                <div className="relative group">
                  <Lock className="absolute left-5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-300 group-focus-within:text-medic-primary transition-colors" />
                  <input
                    type="password"
                    name="password"
                    required
                    value={formData.password}
                    onChange={handleChange}
                    placeholder="••••••••••••"
                    className="w-full pl-14 pr-6 py-5 bg-slate-50 border-2 border-transparent focus:border-medic-primary/20 focus:bg-white rounded-[1.5rem] text-sm font-bold transition-all outline-none"
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-[#0F172A] text-white py-6 rounded-[1.5rem] font-black text-xs uppercase tracking-[0.2em] shadow-2xl shadow-slate-900/20 hover:translate-y-[-2px] hover:shadow-medic-primary/10 active:scale-[0.98] disabled:opacity-70 disabled:pointer-events-none flex items-center justify-center gap-3 transition-all"
              >
                {loading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <>
                    Authorize Access
                    <ArrowRight className="w-5 h-5" />
                  </>
                )}
              </button>
            </form>
          </div>
        </div>

        <p className="text-center mt-10 text-[10px] font-black text-slate-400 uppercase tracking-widest leading-loose">
          MediScan Secure Node • High Authority Zone
          <br />
          Unauthorized Access is Logged and Monitored &copy; 2026
        </p>
      </motion.div>
    </div>
  );
};

export default AdminLogin;

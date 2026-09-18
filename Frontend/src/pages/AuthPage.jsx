import { useState, useEffect } from "react";
import { HeartPulse, ArrowRight, Mail, Lock, Eye, EyeOff } from "lucide-react";
import { useNavigate, useLocation, Link } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { toast } from "react-toastify";
import { useGoogleLogin } from "@react-oauth/google";
import { motion, AnimatePresence } from "framer-motion";
import {
  registerUser,
  loginUser,
  googleLogin,
} from "../store/slices/authSlice";

const AuthPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const dispatch = useDispatch();
  const { loading, isVerifying, isAuthenticated, user } = useSelector(
    (state) => state.auth,
  );

  const isLogin = !location.pathname.includes("signup");
  const [role, setRole] = useState("patient");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  useEffect(() => {
    if (isVerifying) {
      navigate("/verify-otp");
      return;
    }

    if (isAuthenticated && user) {
      if (user.role === "ADMIN" || user.is_superuser) {
        navigate("/admin/dashboard");
      } else if (user.role === "DOCTOR") {
        navigate("/doctor-dashboard");
      } else {
        navigate("/dashboard");
      }
    }
  }, [isVerifying, isAuthenticated, user, navigate]);

  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    confirmPassword: "",
    specialization: "",
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (isLogin) {
      dispatch(
        loginUser({
          email: formData.email,
          password: formData.password,
          role: role.toUpperCase(),
        }),
      );
    } else {
      if (formData.password !== formData.confirmPassword) {
        toast.error("Passwords do not match!");
        return;
      }

      const signupData = {
        first_name: formData.firstName,
        last_name: formData.lastName,
        email: formData.email,
        phone_number: "0000000000",
        password: formData.password,
        confirm_password: formData.confirmPassword,
        role: role.toUpperCase(),
        specialization: role === "doctor" ? formData.specialization : null,
      };
      dispatch(registerUser(signupData));
    }
  };

  const loginWithGoogle = useGoogleLogin({
    onSuccess: (tokenResponse) => {
      dispatch(googleLogin(tokenResponse.access_token))
        .unwrap()
        .then((response) => {
          const userRole = response.user.role;
          if (userRole === "ADMIN") navigate("/admin/dashboard");
          else if (userRole === "DOCTOR") navigate("/doctor-dashboard");
          else navigate("/dashboard");
        });
    },
    onError: () => toast.error("Google Login Failed"),
  });

  return (
    <div className="min-h-screen w-full flex flex-col md:flex-row bg-white overflow-hidden">
      {/* Branding Section (Left) */}
      <div className="md:w-1/2 bg-medic-dark relative overflow-hidden flex flex-col justify-center px-12 md:px-24 py-20 text-white min-h-[40vh] md:min-h-screen">
        {/* Abstract shapes in background */}
        <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-medic-primary/20 rounded-full blur-[120px] -mr-20 -mt-20 pointer-events-none" />
        <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-white/5 rounded-full blur-[80px] -ml-20 -mb-20 pointer-events-none" />
        <div
          className="absolute inset-0 opacity-10 pointer-events-none"
          style={{
            backgroundImage: "radial-gradient(#fff 1px, transparent 1px)",
            backgroundSize: "30px 30px",
          }}
        />

        <motion.div
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          className="relative z-10 space-y-8"
        >
          <Link to="/" className="inline-flex items-center gap-4 group">
            <div className="w-14 h-14 bg-white rounded-2xl flex items-center justify-center shadow-2xl shadow-black/10 transition-transform group-hover:scale-110">
              <HeartPulse size={30} className="text-medic-dark" />
            </div>
            <span className="text-2xl font-black tracking-tighter uppercase">
              Mediscan
            </span>
          </Link>

          <div className="space-y-4">
            <h1 className="text-5xl md:text-7xl font-black leading-[0.95] tracking-tighter">
              Welcome to <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-medic-primary to-white font-black italic">
                Healthcare
              </span>{" "}
              <br />
              Simplified.
            </h1>
            <p className="text-lg md:text-xl text-white/70 max-w-md font-medium leading-relaxed">
              Connect with specialists, track your health markers, and take
              control of your medical journey with enterprise-grade clinical
              intelligence.
            </p>
          </div>
        </motion.div>
      </div>

      {/* Form Section (Right) */}
      <div className="md:w-1/2 flex items-center justify-center p-8 md:p-16 lg:p-24 bg-white z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-md space-y-10"
        >
          <div className="space-y-3">
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-medic-soft text-medic-dark rounded-full mb-2">
              <span className="w-1.5 h-1.5 bg-medic-dark rounded-full animate-pulse" />
              <span className="text-[10px] font-black uppercase tracking-widest">
                {isLogin ? "Login Access" : "New Account"}
              </span>
            </div>
            <h2 className="text-4xl font-black text-slate-900 tracking-tight">
              {isLogin ? "User Login" : "Create Account"}
            </h2>
            <p className="text-slate-400 font-medium">
              {isLogin
                ? "Welcome back! Please enter your details."
                : "Join our network of proactive health users."}
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <AnimatePresence mode="wait">
              {!isLogin && (
                <motion.div
                  key="signup-fields"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="grid grid-cols-1 md:grid-cols-2 gap-4 overflow-hidden"
                >
                  <div className="space-y-2">
                    <label className="text-[11px] font-black text-slate-400 uppercase tracking-widest ml-1">
                      First Name
                    </label>
                    <input
                      type="text"
                      name="firstName"
                      required
                      placeholder=" Pratik"
                      className="w-full px-6 py-4 bg-slate-50 border border-slate-100 rounded-2xl text-base text-slate-900 outline-none focus:border-medic-dark focus:ring-4 focus:ring-medic-dark/5 focus:bg-white transition-all font-bold placeholder:font-medium"
                      onChange={handleChange}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-[11px] font-black text-slate-400 uppercase tracking-widest ml-1">
                      Last Name
                    </label>
                    <input
                      type="text"
                      name="lastName"
                      required
                      placeholder="Chapagain"
                      className="w-full px-6 py-4 bg-slate-50 border border-slate-100 rounded-2xl text-base text-slate-900 outline-none focus:border-medic-dark focus:ring-4 focus:ring-medic-dark/5 focus:bg-white transition-all font-bold placeholder:font-medium"
                      onChange={handleChange}
                    />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="space-y-2">
              <label className="text-[11px] font-black text-slate-400 uppercase tracking-widest ml-1">
                Email Address
              </label>
              <div className="relative">
                <Mail
                  className="absolute left-6 top-1/2 -translate-y-1/2 text-slate-300"
                  size={18}
                />
                <input
                  type="email"
                  name="email"
                  required
                  placeholder="name@example.com"
                  className="w-full pl-14 pr-6 py-4 bg-slate-50 border border-slate-100 rounded-2xl text-base text-slate-900 outline-none focus:border-medic-dark focus:ring-4 focus:ring-medic-dark/5 focus:bg-white transition-all font-bold placeholder:font-medium"
                  onChange={handleChange}
                />
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between items-center ml-1">
                <label className="text-[11px] font-black text-slate-400 uppercase tracking-widest">
                  Password
                </label>
                {isLogin && (
                  <a
                    href="#"
                    className="text-[11px] font-black text-medic-dark hover:underline uppercase tracking-widest"
                  >
                    Forgot?
                  </a>
                )}
              </div>
              <div className="relative">
                <Lock
                  className="absolute left-6 top-1/2 -translate-y-1/2 text-slate-300"
                  size={18}
                />
                <input
                  type={showPassword ? "text" : "password"}
                  name="password"
                  required
                  placeholder="********"
                  className="w-full pl-14 pr-14 py-4 bg-slate-50 border border-slate-100 rounded-2xl text-base text-slate-900 outline-none focus:border-medic-dark focus:ring-4 focus:ring-medic-dark/5 focus:bg-white transition-all font-bold placeholder:font-medium"
                  onChange={handleChange}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-6 top-1/2 -translate-y-1/2 text-slate-300 hover:text-medic-dark transition-colors"
                >
                  {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>

            <AnimatePresence>
              {!isLogin && (
                <motion.div
                  key="confirm-password-field"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="space-y-2 overflow-hidden"
                >
                  <label className="text-[11px] font-black text-slate-400 uppercase tracking-widest ml-1">
                    Confirm Password
                  </label>
                  <div className="relative">
                    <input
                      type={showConfirmPassword ? "text" : "password"}
                      name="confirmPassword"
                      required
                      placeholder="********"
                      className="w-full px-6 py-4 pr-14 bg-slate-50 border border-slate-100 rounded-2xl text-base text-slate-900 outline-none focus:border-medic-dark focus:ring-4 focus:ring-medic-dark/5 focus:bg-white transition-all font-bold placeholder:font-medium"
                      onChange={handleChange}
                    />
                    <button
                      type="button"
                      onClick={() =>
                        setShowConfirmPassword(!showConfirmPassword)
                      }
                      className="absolute right-6 top-1/2 -translate-y-1/2 text-slate-300 hover:text-medic-dark transition-colors"
                    >
                      {showConfirmPassword ? (
                        <EyeOff size={18} />
                      ) : (
                        <Eye size={18} />
                      )}
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <AnimatePresence>
              {!isLogin && role === "doctor" && (
                <motion.div
                  key="specialization-field"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="space-y-2 overflow-hidden"
                >
                  <label className="text-[11px] font-black text-slate-400 uppercase tracking-widest ml-1">
                    Specialization
                  </label>
                  <select
                    name="specialization"
                    value={formData.specialization}
                    onChange={handleChange}
                    className="w-full px-6 py-4 bg-slate-50 border border-slate-100 rounded-2xl text-base text-slate-900 outline-none focus:border-medic-dark focus:ring-4 focus:ring-medic-dark/5 focus:bg-white transition-all font-bold appearance-none cursor-pointer"
                  >
                    <option value="" disabled>
                      Select specialization...
                    </option>
                    <option value="CARDIOLOGIST">Cardiology</option>
                    <option value="ENDOCRINOLOGIST">Endocrinology</option>
                    <option value="NEPHROLOGIST">Nephrology</option>
                    <option value="HEPATOLOGIST">Hepatology</option>
                  </select>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="pt-4">
              <div className="flex p-1 bg-slate-50 border border-slate-100 rounded-2xl mb-8">
                <button
                  type="button"
                  onClick={() => setRole("patient")}
                  className={`flex-1 py-3 text-xs font-black uppercase tracking-widest rounded-xl transition-all ${role === "patient" ? "bg-white text-medic-dark shadow-sm border border-slate-100" : "text-slate-400 hover:text-slate-600"}`}
                >
                  Patient
                </button>
                <button
                  type="button"
                  onClick={() => setRole("doctor")}
                  className={`flex-1 py-3 text-xs font-black uppercase tracking-widest rounded-xl transition-all ${role === "doctor" ? "bg-white text-medic-dark shadow-sm border border-slate-100" : "text-slate-400 hover:text-slate-600"}`}
                >
                  Doctor
                </button>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-medic-dark text-white py-5 rounded-2xl font-black text-base shadow-xl shadow-medic-dark/20 hover:bg-medic-primary transition-all active:scale-[0.98] disabled:opacity-50 flex items-center justify-center gap-3 tracking-widest uppercase"
              >
                {loading ? (
                  <div className="w-6 h-6 border-3 border-white/20 border-t-white rounded-full animate-spin" />
                ) : (
                  <>
                    {isLogin ? "Login Now" : "Create Account"}
                    <ArrowRight size={20} />
                  </>
                )}
              </button>
            </div>

            <div className="text-center pt-2">
              <button
                onClick={() => navigate(isLogin ? "/signup" : "/login")}
                className="text-xs font-black text-slate-400 hover:text-medic-dark transition-colors uppercase tracking-[0.2em]"
              >
                {isLogin
                  ? "New to Mediscan? Join here."
                  : "Already a member? Login here."}
              </button>
            </div>

            <div className="relative py-4">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-slate-50"></div>
              </div>
              <div className="relative flex justify-center text-[10px] font-black uppercase tracking-[0.4em] text-slate-200">
                <span className="bg-white px-6">Third-Party Auth</span>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4">
              <button
                type="button"
                onClick={() => loginWithGoogle()}
                className="w-full py-4 border border-slate-100 rounded-2xl font-bold text-sm text-slate-500 hover:bg-slate-50 transition-all flex items-center justify-center gap-4 group"
              >
                <svg
                  className="w-5 h-5 transition-transform group-hover:scale-110"
                  viewBox="0 0 24 24"
                >
                  <path
                    d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                    fill="#4285F4"
                  />
                  <path
                    d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                    fill="#34A853"
                  />
                  <path
                    d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z"
                    fill="#FBBC05"
                  />
                  <path
                    d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                    fill="#EA4335"
                  />
                </svg>
                Continue with Google
              </button>
            </div>
          </form>
        </motion.div>
      </div>
    </div>
  );
};

export default AuthPage;

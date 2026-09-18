import { useState, useEffect } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { HeartPulse, Menu, X } from "lucide-react";
import { useSelector, useDispatch } from "react-redux";
import { logoutUser } from "../../store/slices/authSlice";
import { motion, AnimatePresence } from "framer-motion";

const Navbar = () => {
  const { isAuthenticated, user } = useSelector((state) => state.auth);
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const location = useLocation();
  const [isOpen, setIsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleLogout = () => {
    dispatch(logoutUser());
    navigate("/login");
    setIsOpen(false);
  };

  const isRestrictedDoctor =
    user?.role === "DOCTOR" && user?.doctor_status !== "VERIFIED";

  const navLinks = [
    { name: "Home", path: "/", show: !isAuthenticated },
    { name: "Doctors", path: "/doctors", show: true },
    { name: "About", path: "/about", show: true },
    { name: "Contact", path: "/contact", show: true },
    { name: "Services", path: "/services", show: true },
    {
      name: "Appointments",
      path: "/appointments",
      show: isAuthenticated && user?.role === "DOCTOR",
      disabled: isRestrictedDoctor,
    },
    {
      name: "Patients",
      path: "/patients",
      show: isAuthenticated && user?.role === "DOCTOR",
      disabled: isRestrictedDoctor,
    },
    {
      name: user?.role === "ADMIN" ? "Admin" : "Dashboard",
      path:
        user?.role === "ADMIN"
          ? "/admin/dashboard"
          : user?.role === "DOCTOR"
            ? "/doctor-dashboard"
            : "/dashboard",
      show: isAuthenticated,
      disabled: isRestrictedDoctor,
    },
  ];

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-[100] transition-all duration-300 ${scrolled ? "bg-white border-b border-slate-200 py-3 shadow-xl shadow-slate-900/5" : "bg-white/50 backdrop-blur-sm py-4"}`}
    >
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center gap-3 z-50 group">
            <div className="w-10 h-10 bg-medic-dark rounded-xl flex items-center justify-center group-hover:bg-medic-primary transition-colors duration-300 shadow-lg shadow-medic-dark/10">
              <HeartPulse className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-black text-slate-900 tracking-tighter">
              MEDISCAN
            </span>
          </Link>

          {/* Desktop Links */}
          <div className="hidden lg:flex items-center gap-6">
            {navLinks
              .filter((link) => link.show)
              .map((link) =>
                link.disabled ? (
                  <div
                    key={link.path}
                    className="relative group cursor-not-allowed"
                  >
                    <span className="text-slate-300 text-sm font-semibold">
                      {link.name}
                    </span>
                  </div>
                ) : (
                  <Link
                    key={link.path}
                    to={link.path}
                    className={`text-base font-black tracking-tight ${location.pathname === link.path ? "text-medic-dark" : "text-slate-500 hover:text-slate-900"} transition-all duration-200`}
                  >
                    {link.name}
                  </Link>
                ),
              )}
          </div>

          <div className="flex items-center gap-4">
            {isAuthenticated ? (
              <div className="hidden md:flex items-center gap-4">
                <Link
                  to="/profile"
                  className="flex items-center gap-2 text-sm font-bold text-slate-700 hover:text-medic-dark transition-colors"
                >
                  <div className="w-8 h-8 rounded-full bg-slate-900 text-white flex items-center justify-center text-[10px] font-black">
                    {user?.first_name?.[0]}
                    {user?.last_name?.[0]}
                  </div>
                  <span>Profile</span>
                </Link>
                <button
                  onClick={handleLogout}
                  className="text-slate-400 hover:text-rose-500 transition-colors text-sm font-bold"
                >
                  Logout
                </button>
              </div>
            ) : (
              <div className="hidden md:flex items-center gap-8">
                <Link
                  to="/login"
                  className="text-base font-bold text-slate-500 hover:text-slate-900 transition-colors"
                >
                  Log in
                </Link>
                <Link
                  to="/signup"
                  className="px-7 py-3 bg-slate-900 text-white text-base font-black rounded-2xl hover:bg-medic-dark transition-all shadow-lg shadow-slate-900/10 active:scale-95"
                >
                  Get Started
                </Link>
              </div>
            )}

            <button
              onClick={() => setIsOpen(!isOpen)}
              className="md:hidden p-2 text-slate-700"
            >
              {isOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsOpen(false)}
              className="fixed inset-0 bg-slate-900/40 z-[80] md:hidden"
            />
            <motion.div
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              className="fixed top-0 right-0 bottom-0 w-64 bg-white z-[90] md:hidden shadow-xl p-8 flex flex-col"
            >
              <div className="flex flex-col gap-6 mt-12">
                {navLinks
                  .filter((link) => link.show)
                  .map((link) => (
                    <Link
                      key={link.name}
                      to={link.path}
                      onClick={() => setIsOpen(false)}
                      className={`text-lg font-medium ${location.pathname === link.path ? "text-medic-dark" : "text-slate-600"}`}
                    >
                      {link.name}
                    </Link>
                  ))}
                <hr className="border-slate-100" />
                {isAuthenticated ? (
                  <>
                    <Link
                      to="/profile"
                      onClick={() => setIsOpen(false)}
                      className="text-lg font-medium text-slate-600"
                    >
                      Profile
                    </Link>
                    <button
                      onClick={handleLogout}
                      className="text-lg font-medium text-rose-500 text-left"
                    >
                      Logout
                    </button>
                  </>
                ) : (
                  <>
                    <Link
                      to="/login"
                      onClick={() => setIsOpen(false)}
                      className="text-lg font-medium text-slate-600"
                    >
                      Login
                    </Link>
                    <Link
                      to="/signup"
                      onClick={() => setIsOpen(false)}
                      className="text-lg font-medium text-medic-dark"
                    >
                      Sign Up
                    </Link>
                  </>
                )}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </nav>
  );
};

export default Navbar;

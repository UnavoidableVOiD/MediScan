import React, { useState } from "react";
import { Outlet, NavLink, useNavigate } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import {
  Users,
  UserPlus,
  Stethoscope,
  LogOut,
  LayoutDashboard,
  Menu,
  X,
  ShieldCheck,
  Bell,
  Search,
  Settings,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { logoutUser } from "../store/slices/authSlice";

const AdminLayout = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { user } = useSelector((state) => state.auth);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const handleLogout = () => {
    dispatch(logoutUser());
    navigate("/admin/login");
  };

  const navItems = [
    { path: "/admin/dashboard", icon: LayoutDashboard, label: "Overview" },
    { path: "/admin/doctors", icon: Stethoscope, label: "Medical Staff" },
    { path: "/admin/patients", icon: Users, label: "Patient Registry" },
  ];

  if (user?.is_superuser) {
    navItems.push({
      path: "/admin/create-admin",
      icon: UserPlus,
      label: "System Access",
    });
  }

  return (
    <div className="flex h-screen bg-[#F8FAFC] overflow-hidden">
      {/* Sidebar */}
      <aside
        className={`${isSidebarOpen ? "w-72" : "w-20"} 
                bg-[#0F172A] text-white transition-all duration-500 ease-in-out fixed inset-y-0 left-0 z-50 md:relative flex flex-col shadow-2xl overflow-hidden`}
      >
        {/* Sidebar Header */}
        <div className="p-8 flex items-center justify-between overflow-hidden whitespace-nowrap">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 bg-medic-primary rounded-xl flex items-center justify-center shadow-lg shadow-medic-primary/20 flex-shrink-0">
              <ShieldCheck className="w-6 h-6 text-[#0F172A]" />
            </div>
            <AnimatePresence>
              {isSidebarOpen && (
                <motion.div
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  className="flex flex-col"
                >
                  <span className="font-black text-xl tracking-tight leading-none">
                    MEDISCAN
                  </span>
                  <span className="text-[10px] font-bold text-medic-primary uppercase tracking-[0.2em] mt-1">
                    Admin Central
                  </span>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          <button
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors hidden md:block"
          >
            {isSidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>

        {/* Nav Links */}
        <nav className="flex-1 px-4 py-6 space-y-2">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center gap-4 px-5 py-4 rounded-2xl transition-all group relative overflow-hidden ${
                  isActive
                    ? "bg-medic-primary text-[#0F172A] font-black shadow-lg shadow-medic-primary/20"
                    : "text-slate-400 hover:bg-white/5 hover:text-white"
                }`
              }
            >
              <item.icon
                className={`w-5 h-5 flex-shrink-0 ${isSidebarOpen ? "" : "mx-auto"}`}
              />
              <AnimatePresence>
                {isSidebarOpen && (
                  <motion.span
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -10 }}
                    className="whitespace-nowrap"
                  >
                    {item.label}
                  </motion.span>
                )}
              </AnimatePresence>
            </NavLink>
          ))}
        </nav>

        {/* Sidebar Footer */}
        <div className="p-6 border-t border-white/5">
          <div
            className={`p-4 rounded-2xl bg-white/5 flex items-center gap-4 transition-all ${isSidebarOpen ? "" : "p-2 justify-center"}`}
          >
            <div className="w-10 h-10 rounded-full bg-slate-700 flex items-center justify-center font-black text-slate-300 flex-shrink-0">
              {user?.email?.[0].toUpperCase()}
            </div>
            {isSidebarOpen && (
              <div className="flex-1 min-w-0">
                <p className="text-xs font-black text-white truncate">
                  {user?.email}
                </p>
                <p className="text-[10px] font-bold text-slate-500 uppercase">
                  Administrator
                </p>
              </div>
            )}
          </div>

          <button
            onClick={handleLogout}
            className={`mt-4 flex items-center gap-4 w-full px-5 py-4 text-red-400 hover:bg-red-500/10 hover:text-red-300 rounded-2xl transition-all group overflow-hidden ${isSidebarOpen ? "" : "justify-center"}`}
          >
            <LogOut className="w-5 h-5 flex-shrink-0 group-hover:-translate-x-1 transition-transform" />
            {isSidebarOpen && (
              <span className="font-bold text-sm">Sign Out</span>
            )}
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Header */}
        <header className="h-20 bg-white border-b border-slate-100 flex items-center justify-between px-8 z-40">
          <div className="flex items-center gap-6 flex-1">
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="md:hidden p-2 text-slate-900"
            >
              <Menu size={24} />
            </button>

            <div className="relative max-w-md w-full hidden sm:block">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                type="text"
                placeholder="Search everything..."
                className="w-full pl-11 pr-4 py-2.5 bg-slate-50 border-none rounded-xl text-sm focus:ring-2 focus:ring-medic-primary/20 transition-all outline-none"
              />
            </div>
          </div>

          <div className="flex items-center gap-4">
            <button className="p-2.5 text-slate-500 hover:bg-slate-50 rounded-xl transition-all relative">
              <Bell size={20} />
              <span className="absolute top-2.5 right-2.5 w-2 h-2 bg-red-500 rounded-full border-2 border-white"></span>
            </button>
            <button className="p-2.5 text-slate-500 hover:bg-slate-50 rounded-xl transition-all">
              <Settings size={20} />
            </button>
            <div className="w-px h-6 bg-slate-200 mx-2 hidden sm:block"></div>
            <div className="hidden sm:flex items-center gap-3">
              <div className="text-right">
                <p className="text-xs font-black text-slate-900">System Root</p>
                <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                  Global Admin
                </p>
              </div>
            </div>
          </div>
        </header>

        <div className="flex-1 overflow-auto p-8 custom-scrollbar">
          <Outlet />
        </div>
      </main>
    </div>
  );
};

export default AdminLayout;

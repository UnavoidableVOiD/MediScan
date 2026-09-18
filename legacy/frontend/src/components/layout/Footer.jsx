import React from "react";
import { HeartPulse, Github, Twitter, Linkedin, Mail } from "lucide-react";
import { Link } from "react-router-dom";

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-white border-t border-slate-100 pt-24 pb-12 overflow-hidden">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid grid-cols-2 md:grid-cols-12 gap-12 mb-20">
          {/* Brand & Mission */}
          <div className="col-span-2 md:col-span-4">
            <Link to="/" className="flex items-center gap-2 mb-8 group">
              <HeartPulse className="w-8 h-8 text-medic-dark group-hover:text-medic-accent transition-colors" />
              <span className="text-2xl font-black tracking-tighter text-slate-900">
                Mediscan
              </span>
            </Link>
            <p className="text-slate-500 font-medium leading-relaxed max-w-sm mb-8 italic">
              Empowering individuals to understand their health story through
              intelligent interpretation and clinical connectivity.
            </p>
            <div className="flex gap-4">
              {[Twitter, Github, Linkedin].map((Icon, i) => (
                <a
                  key={i}
                  href="#"
                  className="w-10 h-10 rounded-full bg-slate-50 flex items-center justify-center text-slate-400 hover:bg-medic-dark hover:text-white transition-all transform hover:-translate-y-1"
                >
                  <Icon size={18} />
                </a>
              ))}
            </div>
          </div>

          {/* Navigation Links */}
          <div className="col-span-1 md:col-span-2 md:col-start-6">
            <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-900 mb-8">
              Platform
            </h4>
            <ul className="space-y-4">
              <li>
                <Link
                  to="/doctors"
                  className="text-sm font-bold text-slate-500 hover:text-medic-dark transition-colors"
                >
                  Specialists
                </Link>
              </li>
              <li>
                <Link
                  to="/login"
                  className="text-sm font-bold text-slate-500 hover:text-medic-dark transition-colors"
                >
                  Analysis
                </Link>
              </li>
              <li>
                <Link
                  to="/signup"
                  className="text-sm font-bold text-slate-500 hover:text-medic-dark transition-colors"
                >
                  Registration
                </Link>
              </li>
            </ul>
          </div>

          <div className="col-span-1 md:col-span-2">
            <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-900 mb-8">
              Company
            </h4>
            <ul className="space-y-4">
              <li>
                <Link
                  to="#"
                  className="text-sm font-bold text-slate-500 hover:text-medic-dark transition-colors"
                >
                  About Us
                </Link>
              </li>
              <li>
                <Link
                  to="#"
                  className="text-sm font-bold text-slate-500 hover:text-medic-dark transition-colors"
                >
                  Our Ethos
                </Link>
              </li>
              <li>
                <Link
                  to="#"
                  className="text-sm font-bold text-slate-500 hover:text-medic-dark transition-colors"
                >
                  Careers
                </Link>
              </li>
            </ul>
          </div>

          <div className="col-span-2 md:col-span-3">
            <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-900 mb-8">
              Newsletter
            </h4>
            <p className="text-sm text-slate-500 font-medium mb-6">
              Stay informed about health tech updates.
            </p>
            <div className="relative">
              <input
                type="email"
                placeholder="name@company.com"
                className="w-full px-5 py-4 bg-slate-50 border border-slate-100 rounded-2xl text-sm font-medium focus:outline-none focus:border-medic-dark transition-all"
              />
              <button className="absolute right-2 top-2 bottom-2 px-4 bg-medic-dark text-white rounded-xl text-xs font-black uppercase tracking-widest hover:bg-medic-primary transition-all">
                Join
              </button>
            </div>
          </div>
        </div>

        <div className="pt-12 border-t border-slate-50 flex flex-col md:flex-row justify-between items-center gap-6">
          <p className="text-xs font-black uppercase tracking-widest text-slate-400">
            &copy; {currentYear} Mediscan. Built for Clinical Clarity.
          </p>
          <div className="flex gap-8">
            <Link
              to="#"
              className="text-[10px] font-black uppercase tracking-widest text-slate-400 hover:text-slate-900"
            >
              Privacy Policy
            </Link>
            <Link
              to="#"
              className="text-[10px] font-black uppercase tracking-widest text-slate-400 hover:text-slate-900"
            >
              Terms of Service
            </Link>
            <Link
              to="#"
              className="text-[10px] font-black uppercase tracking-widest text-slate-400 hover:text-slate-900"
            >
              Cookie Protocol
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;

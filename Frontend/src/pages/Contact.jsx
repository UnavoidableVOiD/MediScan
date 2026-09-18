import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  Send,
  Mail,
  Phone,
  MapPin,
  MessageCircle,
  Loader2,
  Sparkles,
} from "lucide-react";
import { toast } from "react-toastify";

const Contact = () => {
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    message: "",
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setLoading(true);

    // Simulate API call
    setTimeout(() => {
      setLoading(false);
      toast.success("Your message has been sent successfully!");
      setFormData({ name: "", email: "", message: "" });
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-neutral-background pt-32 pb-20 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-20 space-y-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 px-4 py-1.5 bg-medic-dark/5 rounded-full text-medic-dark font-bold text-xs tracking-widest uppercase"
          >
            <Sparkles className="w-3 h-3 text-medic-dark" />
            Here to Help
          </motion.div>
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-5xl md:text-6xl font-black text-slate-900 tracking-tight"
          >
            Get in <span className="text-medic-dark">Touch.</span>
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-xl text-slate-500 max-w-2xl mx-auto font-medium leading-relaxed"
          >
            We're here to help — reach out anytime with questions, feedback, or
            support requests.
          </motion.p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-16 items-start">
          {/* Contact Form */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-12 xl:col-span-7 bg-white p-10 md:p-14 rounded-[3rem] shadow-2xl shadow-medic-dark/5 border border-slate-100"
          >
            <form onSubmit={handleSubmit} className="space-y-8">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="space-y-2">
                  <label className="text-xs font-black text-slate-400 uppercase tracking-widest block ml-1">
                    Your Name
                  </label>
                  <input
                    type="text"
                    name="name"
                    required
                    value={formData.name}
                    onChange={handleChange}
                    placeholder="Supreme Badal"
                    className="w-full px-8 py-5 bg-neutral-soft rounded-[1.5rem] border-transparent focus:border-medic-dark focus:bg-white transition-all outline-none text-slate-900 font-bold placeholder:font-medium placeholder:text-slate-400 focus:ring-4 focus:ring-medic-dark/5"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-black text-slate-400 uppercase tracking-widest block ml-1">
                    Email Address
                  </label>
                  <input
                    type="email"
                    name="email"
                    required
                    value={formData.email}
                    onChange={handleChange}
                    placeholder="aashish@example.com"
                    className="w-full px-8 py-5 bg-neutral-soft rounded-[1.5rem] border-transparent focus:border-medic-dark focus:bg-white transition-all outline-none text-slate-900 font-bold placeholder:font-medium placeholder:text-slate-400 focus:ring-4 focus:ring-medic-dark/5"
                  />
                </div>
              </div>
              <div className="space-y-2">
                <label className="text-xs font-black text-slate-400 uppercase tracking-widest block ml-1">
                  Your Message
                </label>
                <textarea
                  name="message"
                  required
                  rows="6"
                  value={formData.message}
                  onChange={handleChange}
                  placeholder="How can we help you?"
                  className="w-full px-8 py-5 bg-neutral-soft rounded-[2rem] border-transparent focus:border-medic-dark focus:bg-white transition-all outline-none text-slate-900 font-bold placeholder:font-medium placeholder:text-slate-400 focus:ring-4 focus:ring-medic-dark/5 resize-none"
                ></textarea>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="group w-full md:w-auto px-12 py-5 bg-medic-dark text-white rounded-[1.5rem] font-black tracking-wide text-sm shadow-xl shadow-medic-dark/20 hover:bg-medic-primary transition-all active:scale-95 disabled:opacity-50 flex items-center justify-center gap-3 hover:-translate-y-1"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Sending Request...
                  </>
                ) : (
                  <>
                    <Send className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                    Send Message
                  </>
                )}
              </button>
            </form>
          </motion.div>

          {/* Support Info Panel */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-12 xl:col-span-5 space-y-8"
          >
            <div className="bg-medic-dark p-12 rounded-[3rem] text-white overflow-hidden relative shadow-2xl shadow-medic-dark/20">
              <MessageCircle className="absolute -bottom-10 -right-10 w-64 h-64 text-white/5 rotate-12" />
              <h3 className="text-3xl font-black mb-10 relative z-10 tracking-tight">
                Direct Support
              </h3>

              <div className="space-y-8 relative z-10">
                <div className="flex items-start gap-6 group">
                  <div className="w-14 h-14 bg-white/10 rounded-[1.2rem] flex items-center justify-center flex-shrink-0 group-hover:bg-medic-accent group-hover:text-medic-dark transition-colors">
                    <Mail className="w-7 h-7 text-medic-light group-hover:text-medic-dark transition-colors" />
                  </div>
                  <div>
                    <p className="text-xs font-black text-medic-light/50 uppercase tracking-widest mb-1">
                      Email Us
                    </p>
                    <p className="text-xl font-bold">support@mediscan.com</p>
                  </div>
                </div>

                <div className="flex items-start gap-6 group">
                  <div className="w-14 h-14 bg-white/10 rounded-[1.2rem] flex items-center justify-center flex-shrink-0 group-hover:bg-medic-accent group-hover:text-medic-dark transition-colors">
                    <Phone className="w-7 h-7 text-medic-light group-hover:text-medic-dark transition-colors" />
                  </div>
                  <div>
                    <p className="text-xs font-black text-medic-light/50 uppercase tracking-widest mb-1">
                      Call Us
                    </p>
                    <p className="text-xl font-bold">+977 9800000000</p>
                  </div>
                </div>

                <div className="flex items-start gap-6 group">
                  <div className="w-14 h-14 bg-white/10 rounded-[1.2rem] flex items-center justify-center flex-shrink-0 group-hover:bg-medic-accent group-hover:text-medic-dark transition-colors">
                    <MapPin className="w-7 h-7 text-medic-light group-hover:text-medic-dark transition-colors" />
                  </div>
                  <div>
                    <p className="text-xs font-black text-medic-light/50 uppercase tracking-widest mb-1">
                      Address
                    </p>
                    <p className="text-xl font-bold leading-tight">
                      Koteshwor-32, <br />
                      Kathmandu, Nepal
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white p-12 rounded-[3rem] border border-slate-100 shadow-xl shadow-slate-200/50">
              <h4 className="font-black text-2xl text-slate-900 mb-4 tracking-tight">
                Need answers fast?
              </h4>
              <p className="text-slate-500 mb-8 font-medium leading-relaxed">
                Check our FAQ page for common questions about report analysis,
                account security, and data privacy.
              </p>
              <a
                href="/faq"
                className="inline-flex items-center gap-2 text-medic-dark font-black hover:gap-4 transition-all"
              >
                Read FAQ <span className="text-xl">&rarr;</span>
              </a>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Contact;

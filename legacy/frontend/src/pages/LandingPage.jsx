import {
  ArrowRight,
  FileText,
  Activity,
  CheckCircle2,
  Upload,
  Brain,
  Stethoscope,
  Shield,
  Search,
  MessageSquare,
} from "lucide-react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";

const LandingPage = () => {
  return (
    <div className="bg-white overflow-hidden scroll-smooth">
      {/* Hero Section */}
      <section className="relative min-h-[95vh] flex items-center pt-32 pb-20 bg-slate-50/80">
        <div className="absolute inset-0 z-0 opacity-30 pointer-events-none">
          <div
            className="absolute inset-0"
            style={{
              backgroundImage: "radial-gradient(#94a3b8 1px, transparent 1px)",
              backgroundSize: "40px 40px",
            }}
          />
        </div>

        {/* Subtle Gradient Blobs */}
        <div className="absolute top-[10%] right-[5%] w-[45%] h-[45%] bg-blue-100/40 rounded-full blur-[120px] animate-pulse" />
        <div className="absolute bottom-[10%] left-[5%] w-[40%] h-[40%] bg-medic-soft/30 rounded-full blur-[120px]" />

        <div className="max-w-7xl mx-auto px-6 relative z-10 w-full">
          <div className="grid lg:grid-cols-2 gap-20 items-center">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
            >
              <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-white border border-slate-200 rounded-full mb-8 shadow-sm">
                <span className="flex h-2 w-2 rounded-full bg-medic-primary animate-pulse" />
                <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">
                  Refined Healthcare Intelligence
                </span>
              </div>

              <h1 className="text-6xl md:text-8xl font-black mb-8 leading-[0.95] tracking-tighter text-slate-900">
                Understand <br />
                Your <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-medic-dark to-medic-primary">
                  Report Story.
                </span>
              </h1>

              <p className="text-xl md:text-2xl text-slate-500 font-medium leading-relaxed max-w-xl mb-12">
                Bridge the gap between complex diagnostic data and clear
                clinical insights. Navigate your health journey with
                professional confidence.
              </p>

              <div className="flex flex-col sm:flex-row gap-5">
                <Link
                  to="/signup"
                  className="px-10 py-5 bg-medic-dark text-white rounded-2xl font-bold text-lg shadow-xl shadow-medic-dark/20 hover:-translate-y-1 transition-all flex items-center justify-center gap-3"
                >
                  Get Started Free <ArrowRight size={22} />
                </Link>
                <Link
                  to="/doctors"
                  className="px-10 py-5 bg-white text-slate-900 border border-slate-200 rounded-2xl font-bold text-lg hover:bg-slate-50 transition-all text-center"
                >
                  View Specialists
                </Link>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 1, delay: 0.2 }}
              className="relative hidden lg:block"
            >
              <div className="bg-white rounded-[3rem] p-10 shadow-2xl shadow-slate-200 border border-slate-100">
                <div className="flex items-center justify-between mb-8 pb-6 border-b border-slate-50">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-medic-soft rounded-2xl flex items-center justify-center">
                      <FileText className="text-medic-dark" size={24} />
                    </div>
                    <div>
                      <h4 className="font-bold text-slate-900 text-lg">
                        Lab Report Parse
                      </h4>
                      <p className="text-xs font-medium text-slate-400">
                        Processing Diagnostic Data
                      </p>
                    </div>
                  </div>
                  <CheckCircle2 className="text-medic-primary" size={24} />
                </div>
                <div className="space-y-4">
                  <div className="h-4 w-full bg-slate-50 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-medic-primary"
                      initial={{ width: 0 }}
                      animate={{ width: "75%" }}
                      transition={{ duration: 1.5, delay: 0.5 }}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="h-20 bg-slate-50 rounded-2xl p-4 flex flex-col justify-center">
                      <p className="text-[10px] font-bold text-slate-400 uppercase">
                        Glucose
                      </p>
                      <p className="text-xl font-black text-slate-900">
                        94 mg/dL
                      </p>
                    </div>
                    <div className="h-20 bg-slate-50 rounded-2xl p-4 flex flex-col justify-center border-l-4 border-medic-primary">
                      <p className="text-[10px] font-bold text-slate-400 uppercase">
                        Hemoglobin
                      </p>
                      <p className="text-xl font-black text-slate-900">
                        14.2 g/dL
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="absolute -z-10 -bottom-6 -right-6 w-full h-full border border-slate-200 rounded-[3rem]" />
            </motion.div>
          </div>
        </div>
      </section>

      {/* Narrative Section - White Section */}
      <section className="py-32 bg-white">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <span className="text-[10px] font-bold text-medic-dark uppercase tracking-[0.4em] mb-6 block">
              Our Mission
            </span>
            <h2 className="text-4xl md:text-5xl font-black text-slate-900 mb-10 tracking-tight leading-tight">
              Giving You Clarity Amidst <br /> Complex Medical Jargon.
            </h2>
            <p className="text-xl text-slate-500 leading-relaxed font-medium">
              Staring at lab results shouldn&apos;t feel like deciphering code.
              Mediscan provides the interpretative bridge, helping you
              understand what your reports mean so you can have smarter
              conversations with your doctors.
            </p>
          </motion.div>
        </div>
      </section>

      {/* How it Works - Dark Section */}
      <section className="py-32 bg-medic-dark rounded-[4rem] mx-6 my-10 overflow-hidden relative">
        <div
          className="absolute inset-0 opacity-10"
          style={{
            backgroundImage: "radial-gradient(#fff 1px, transparent 1px)",
            backgroundSize: "30px 30px",
          }}
        />
        <div className="max-w-7xl mx-auto px-10 relative z-10 text-white">
          <div className="mb-20">
            <h2 className="text-4xl md:text-6xl font-black tracking-tighter">
              How Mediscan Works.
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-16">
            {[
              {
                step: "01",
                title: "Upload Report",
                desc: "Securely upload your PDF or image-based diagnostics.",
                icon: Upload,
              },
              {
                step: "02",
                title: "Smart Analysis",
                desc: "Our engine extracts and interprets key health markers.",
                icon: Brain,
              },
              {
                step: "03",
                title: "Expert Connect",
                desc: "Discuss findings with verified medical specialists.",
                icon: Stethoscope,
              },
            ].map((item, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.2 }}
                className="group"
              >
                <div className="w-16 h-16 bg-white/10 rounded-2xl flex items-center justify-center mb-8 border border-white/10 group-hover:bg-medic-primary transition-all duration-300">
                  <item.icon size={28} />
                </div>
                <p className="text-[10px] font-bold text-white/40 uppercase tracking-widest mb-2">
                  Step {item.step}
                </p>
                <h3 className="text-2xl font-bold mb-4">{item.title}</h3>
                <p className="text-white/60 font-medium leading-relaxed">
                  {item.desc}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Feature Grid - Slate-50 Section */}
      <section className="py-32 bg-slate-50 border-y border-slate-200/60">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col md:flex-row md:items-end justify-between mb-20 gap-8">
            <div className="max-w-2xl">
              <h2 className="text-4xl md:text-5xl font-black text-slate-900 tracking-tight leading-tight mb-6">
                Built for Accuracy <br /> and Privacy.
              </h2>
              <div className="h-1.5 w-20 bg-medic-primary rounded-full" />
            </div>
            <p className="text-slate-500 font-medium max-w-sm italic">
              &quot;Advanced clinical intelligence designed to put the power
              back in your hands.&quot;
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              {
                icon: Shield,
                title: "Secure Vault",
                desc: "Enterprise-grade encryption for all your diagnostic history.",
              },
              {
                icon: Activity,
                title: "Trend Tracker",
                desc: "Visualize markers over time to see how your health evolves.",
              },
              {
                icon: Search,
                title: "Specialist Search",
                desc: "Find verified doctors across multiple disciplines easily.",
              },
              {
                icon: MessageSquare,
                title: "Direct Chat",
                desc: "Communicate with specialists through our secure portal.",
              },
            ].map((f, i) => (
              <motion.div
                key={i}
                className="p-8 bg-white border border-slate-100 rounded-3xl hover:shadow-xl hover:-translate-y-1 transition-all"
              >
                <div className="w-12 h-12 bg-medic-soft rounded-xl flex items-center justify-center text-medic-dark mb-6">
                  <f.icon size={24} />
                </div>
                <h3 className="text-lg font-bold text-slate-900 mb-3">
                  {f.title}
                </h3>
                <p className="text-slate-500 text-sm font-medium leading-relaxed">
                  {f.desc}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Specialist Teaser - White Section */}
      <section className="py-32 bg-white">
        <div className="max-w-7xl mx-auto px-6">
          <div className="bg-medic-dark rounded-[3rem] p-10 md:p-20 flex flex-col lg:flex-row items-center gap-16 relative overflow-hidden text-white shadow-2xl">
            <div className="flex-1 relative z-10">
              <h2 className="text-4xl md:text-6xl font-black mb-8 leading-tight tracking-tight">
                Verified Expertise <br /> Within Reach.
              </h2>
              <p className="text-xl text-slate-400 mb-12 font-medium">
                Our platform hosts a curated network of licensed medical
                professionals, ensuring you get the expert opinion you deserve.
              </p>
              <Link
                to="/doctors"
                className="inline-flex items-center gap-3 font-bold text-medic-primary hover:text-white transition-all text-xl"
              >
                Browse Specialists <ArrowRight size={24} />
              </Link>
            </div>
            <div className="flex-1 grid grid-cols-2 gap-4 relative z-10">
              {["Cardiology", "Nephrology", "Endocrinology", "Hepatology"].map(
                (tag) => (
                  <div
                    key={tag}
                    className="p-8 bg-white/5 border border-white/10 backdrop-blur-md rounded-2xl text-center hover:bg-white/10 transition-all"
                  >
                    <p className="text-xl font-bold">{tag}</p>
                  </div>
                ),
              )}
            </div>
            <div className="absolute top-0 right-0 w-[400px] h-[400px] bg-medic-primary/20 rounded-full blur-[120px] -mr-20 -mt-20" />
          </div>
        </div>
      </section>

      {/* CTA Section - Slate-50 Section */}
      <section className="py-32 bg-slate-50 border-t border-slate-200/60">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
          >
            <h2 className="text-5xl md:text-7xl font-black text-slate-900 mb-10 tracking-tighter">
              Your Health is <br /> Your Wealth.
            </h2>
            <Link
              to="/signup"
              className="inline-block px-12 py-6 bg-medic-dark text-white rounded-2xl font-bold text-xl shadow-2xl shadow-medic-dark/20 hover:scale-[1.02] active:scale-95 transition-all mb-8"
            >
              Start Free Analysis
            </Link>
            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
              Join 12,000+ proactive individuals today
            </p>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default LandingPage;

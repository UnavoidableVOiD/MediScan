import React from "react";
import { motion } from "framer-motion";
import {
  HeartPulse,
  ShieldCheck,
  Cpu,
  Heart,
  Users,
  Globe,
  Target,
  Zap,
} from "lucide-react";

const About = () => {
  const sections = [
    {
      icon: Target,
      title: "Purpose & Vision",
      content:
        "Mediscan was born from a simple yet powerful idea: making complex medical data understandable for everyone. We envision a world where patients are empowered with clear, AI-driven insights into their health.",
      color: "text-blue-600",
      bg: "bg-blue-50",
    },
    {
      icon: Cpu,
      title: "AI in Healthcare",
      content:
        "Our AI engine is trained to identify and explain key health indicators from medical reports. By automating the extraction and summary of data, we reduce the cognitive load for both patients and providers.",
      color: "text-medic-dark",
      bg: "bg-medic-light/20",
    },
    {
      icon: ShieldCheck,
      title: "Ethical AI",
      content:
        "Privacy and fairness are at the core of Mediscan. We use state-of-the-art encryption (HIPAA-aligned) and ensure our AI explanations are informational transparent and safe.",
      color: "text-medic-accent",
      bg: "bg-medic-accent/10",
    },
  ];

  return (
    <div className="flex flex-col w-full overflow-x-hidden">
      {/* Hero Section */}
      <section className="relative py-32 overflow-hidden bg-medic-dark text-white">
        {/* Background Elements */}
        <div className="absolute top-0 right-0 w-[60vw] h-[60vh] bg-medic-primary/20 rounded-full blur-[120px] translate-x-1/2 -translate-y-1/2" />
        <div className="absolute bottom-0 left-0 w-[50vw] h-[50vh] bg-medic-accent/10 rounded-full blur-[100px] -translate-x-1/2 translate-y-1/2" />

        <div className="max-w-7xl mx-auto px-6 relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="max-w-4xl"
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/10 border border-white/20 rounded-full text-white/90 font-bold text-xs tracking-widest uppercase backdrop-blur-md mb-8">
              <HeartPulse className="w-3 h-3 text-medic-accent" />
              Our Mission
            </div>
            <h1 className="text-5xl md:text-7xl font-black mb-8 tracking-tight leading-tight">
              Empowering healthcare with{" "}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-medic-accent to-medic-light">
                AI insights.
              </span>
            </h1>
            <p className="text-xl md:text-2xl text-medic-light/80 leading-relaxed max-w-2xl font-medium">
              Bridging the gap between complex medical data and actionable
              health knowledge for everyone.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Project Overview */}
      <section className="py-32 max-w-7xl mx-auto px-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-20 items-center">
          <div>
            <h2 className="text-4xl font-black text-slate-900 mb-8 flex items-center gap-4">
              <span className="w-2 h-10 bg-medic-dark rounded-full"></span>
              What is Mediscan?
            </h2>
            <div className="space-y-6 text-lg text-slate-600 leading-relaxed font-medium">
              <p>
                Mediscan is an intelligent health assistant that simplifies
                medical reports. We believe that medical knowledge shouldn't be
                locked behind complex terminology and dense data tables.
              </p>
              <p>
                Our platform uses advanced Optical Character Recognition (OCR)
                and Natural Language Processing (NLP) to read your lab results
                and generate easy-to-understand summaries. Whether you're a
                patient tracking chronic health or a doctor needing a quick
                summary, Mediscan is built for you.
              </p>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-8">
            <motion.div
              whileHover={{ y: -10 }}
              className="bg-neutral-soft p-10 rounded-[2.5rem] space-y-6"
            >
              <div className="bg-white w-16 h-16 rounded-2xl flex items-center justify-center shadow-lg shadow-gray-200/50">
                <Zap className="w-8 h-8 text-medic-dark" />
              </div>
              <h4 className="font-black text-2xl text-slate-900">
                Fast & Efficient
              </h4>
            </motion.div>
            <motion.div
              whileHover={{ y: -10 }}
              className="bg-medic-light/20 p-10 rounded-[2.5rem] space-y-6 mt-12"
            >
              <div className="bg-white w-16 h-16 rounded-2xl flex items-center justify-center shadow-lg shadow-medic-dark/5">
                <Globe className="w-8 h-8 text-medic-accent" />
              </div>
              <h4 className="font-black text-2xl text-slate-900">
                Global Access
              </h4>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Core Values / Specific Sections */}
      <section className="bg-neutral-soft/30 py-32">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {sections.map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ y: -10 }}
                className="bg-white p-10 rounded-[2.5rem] shadow-xl shadow-medic-dark/5 border border-white hover:border-medic-dark/20 transition-all"
              >
                <div
                  className={`w-16 h-16 ${item.bg} rounded-3xl flex items-center justify-center mb-8`}
                >
                  <item.icon className={`w-8 h-8 ${item.color}`} />
                </div>
                <h3 className="text-2xl font-black text-slate-900 mb-4">
                  {item.title}
                </h3>
                <p className="text-slate-500 leading-relaxed font-medium">
                  {item.content}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Final CTA/Statement */}
      <section className="py-32 max-w-6xl mx-auto px-6 text-center">
        <div className="bg-medic-dark rounded-[3rem] p-16 md:p-24 text-white overflow-hidden relative shadow-2xl shadow-medic-dark/30">
          <Heart className="absolute -top-10 -right-10 w-96 h-96 text-white/5 animate-pulse" />
          <div className="relative z-10">
            <h2 className="text-4xl md:text-5xl font-black mb-8 tracking-tight">
              Our commitment is to your health.
            </h2>
            <p className="text-medic-light/80 text-xl mb-0 max-w-3xl mx-auto font-medium leading-relaxed">
              We are continuously refining our models and interfaces to ensure
              the highest degree of accuracy and empathy in every medical
              insight we provide.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default About;

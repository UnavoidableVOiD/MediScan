import React from "react";
import { motion } from "framer-motion";
import {
  Brain,
  Stethoscope,
  ShieldCheck,
  Activity,
  ArrowRight,
  CheckCircle2,
  Zap,
  Clock,
} from "lucide-react";
import { useNavigate } from "react-router-dom";

const ServiceCard = ({
  icon: Icon,
  title,
  description,
  benefits,
  delay,
  color,
}) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ duration: 0.5, delay }}
    className="group relative p-8 rounded-[2.5rem] bg-white border border-gray-100 shadow-xl shadow-medic-dark/5 hover:shadow-2xl hover:shadow-medic-dark/10 hover:-translate-y-1 transition-all overflow-hidden"
  >
    <div
      className={`absolute top-0 right-0 w-32 h-32 ${color} opacity-5 rounded-bl-full group-hover:scale-110 transition-transform duration-500`}
    />

    <div
      className={`w-14 h-14 rounded-2xl ${color} bg-opacity-10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}
    >
      <Icon className={`w-7 h-7 ${color.replace("bg-", "text-")}`} />
    </div>

    <h3 className="text-xl font-black text-gray-900 mb-3 tracking-tight">
      {title}
    </h3>

    <p className="text-gray-500 font-medium leading-relaxed mb-6">
      {description}
    </p>

    <ul className="space-y-3">
      {benefits.map((benefit, i) => (
        <li
          key={i}
          className="flex items-center gap-3 text-sm font-bold text-gray-700"
        >
          <CheckCircle2
            size={16}
            className={`flex-shrink-0 ${color.replace("bg-", "text-")}`}
          />
          {benefit}
        </li>
      ))}
    </ul>
  </motion.div>
);

const Services = () => {
  const navigate = useNavigate();

  const services = [
    {
      icon: Brain,
      title: "AI Diagnostic Analysis",
      description:
        "Our core technology uses advanced OCR and machine learning to analyze medical reports instantly, providing detailed insights and risk assessments.",
      benefits: [
        "Instant Report Processing",
        "Key Biomarker Extraction",
        "Risk Level Assessment",
        "Historical Trend Analysis",
      ],
      color: "bg-medic-dark",
      delay: 0.1,
    },
    {
      icon: Stethoscope,
      title: "Specialist Consultation",
      description:
        "Connect seamlessly with verified healthcare providers. Our platform matches you with the right specialists based on your medical profile.",
      benefits: [
        "Verified Doctor Network",
        "Seamless Appointment Booking",
        "Secure Private Messaging",
        "Digital Prescriptions",
      ],
      color: "bg-medic-primary",
      delay: 0.2,
    },
    {
      icon: ShieldCheck,
      title: "Secure Health Vault",
      description:
        "Your health data is sensitive. We treat it that way with enterprise-grade encryption and strict privacy controls compliant with medical standards.",
      benefits: [
        "End-to-End Encryption",
        "Role-Based Access Control",
        "Complete Data Ownership",
        "Audit Logs & History",
      ],
      color: "bg-neutral-dark",
      delay: 0.3,
    },
  ];

  return (
    <div className="min-h-screen bg-neutral-background">
      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-6 overflow-hidden">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-7xl h-full pointer-events-none">
          <div className="absolute top-20 right-0 w-96 h-96 bg-medic-primary/10 rounded-full blur-3xl animate-pulse" />
          <div className="absolute top-40 left-0 w-72 h-72 bg-medic-dark/10 rounded-full blur-3xl animate-pulse delay-1000" />
        </div>

        <div className="max-w-7xl mx-auto relative z-10 text-center space-y-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white border border-gray-100 shadow-sm mb-4"
          >
            <Zap size={16} className="text-medic-primary fill-medic-primary" />
            <span className="text-xs font-black tracking-widest uppercase text-gray-500">
              Powered by Advanced AI
            </span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-5xl md:text-7xl font-black text-gray-900 tracking-tight leading-tight"
          >
            Modern Healthcare, <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-medic-dark to-medic-primary">
              Simplified by Intelligence
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-xl text-gray-500 font-medium max-w-2xl mx-auto leading-relaxed"
          >
            Experience the future of medical care with our integrated platform
            combining AI diagnostics, effortless scheduling, and secure health
            management.
          </motion.p>
        </div>
      </section>

      {/* Services Grid */}
      <section className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {services.map((service, idx) => (
              <ServiceCard key={idx} {...service} />
            ))}
          </div>
        </div>
      </section>

      {/* Feature Highlight / Workflow */}
      <section className="py-20 px-6 relative overflow-hidden">
        <div className="max-w-7xl mx-auto bg-medic-dark rounded-[3rem] p-12 md:p-20 relative overflow-hidden">
          {/* Background Patterns */}
          <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-white/5 rounded-full blur-3xl -mr-32 -mt-32" />
          <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-medic-primary/20 rounded-full blur-3xl -ml-32 -mb-32" />

          <div className="relative z-10 grid grid-cols-1 md:grid-cols-2 gap-16 items-center">
            <div className="space-y-8">
              <h2 className="text-4xl font-black text-white tracking-tight leading-tight">
                From Lab Report to <br />
                Doctor's Plan in Minutes
              </h2>
              <p className="text-lg text-white/80 font-medium leading-relaxed">
                Stop waiting days for clarity. Simply upload your medical
                report, let our AI analyze the critical data, and instantly book
                a consultation with the right specialist.
              </p>

              <div className="space-y-6 pt-4">
                {[
                  {
                    title: "Upload Report",
                    desc: "Securely upload PDF or Image",
                  },
                  {
                    title: "AI Analysis",
                    desc: "Instant extraction of key metrics",
                  },
                  {
                    title: "Connect",
                    desc: "Share results with verified doctors",
                  },
                ].map((step, i) => (
                  <div key={i} className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-2xl bg-white/10 flex items-center justify-center font-black text-white border border-white/10">
                      {i + 1}
                    </div>
                    <div>
                      <h4 className="font-bold text-white text-lg">
                        {step.title}
                      </h4>
                      <p className="text-white/60 text-sm font-medium">
                        {step.desc}
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              <button
                onClick={() => navigate("/login")}
                className="mt-8 px-10 py-5 bg-white text-medic-dark rounded-2xl font-black tracking-wide hover:shadow-2xl hover:scale-105 transition-all flex items-center gap-3"
              >
                START YOUR JOURNEY
                <ArrowRight size={20} />
              </button>
            </div>

            {/* Visual Abstract Representation */}
            <div className="relative hidden md:block">
              <div className="absolute inset-0 bg-gradient-to-br from-medic-primary/20 to-transparent rounded-[2rem] transform rotate-3" />
              <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-[2.5rem] p-8 transform -rotate-2 hover:rotate-0 transition-transform duration-500">
                <div className="space-y-6">
                  {/* Fake UI Elements */}
                  <div className="flex items-center justify-between border-b border-white/10 pb-6">
                    <div className="flex items-center gap-3">
                      <Activity className="text-medic-primary" />
                      <div className="h-2 w-24 bg-white/20 rounded-full" />
                    </div>
                    <div className="h-8 w-8 rounded-full bg-white/20" />
                  </div>
                  <div className="space-y-3">
                    <div className="h-4 w-3/4 bg-white/10 rounded-full" />
                    <div className="h-4 w-1/2 bg-white/10 rounded-full" />
                    <div className="h-32 w-full bg-white/5 rounded-2xl border border-white/10 mt-4" />
                  </div>
                  <div className="flex gap-4 pt-4">
                    <div className="h-10 w-full bg-medic-primary rounded-xl" />
                    <div className="h-10 w-1/3 bg-white/10 rounded-xl" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* FAQ / Support Teaser */}
      <section className="py-20 px-6 bg-white">
        <div className="max-w-4xl mx-auto text-center space-y-10">
          <div className="space-y-4">
            <h2 className="text-4xl font-black text-gray-900 tracking-tight">
              Ready to take control?
            </h2>
            <p className="text-gray-500 font-medium text-lg">
              Join thousands of users who have simplified their healthcare
              journey with Mediscan.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="p-8 bg-neutral-soft rounded-3xl border border-gray-100 text-left cursor-pointer hover:bg-neutral-soft/80 transition-colors">
              <Clock className="w-8 h-8 text-medic-dark mb-4" />
              <h4 className="font-bold text-gray-900 text-lg mb-2">
                24/7 Support
              </h4>
              <p className="text-gray-500 text-sm font-medium">
                Our team is always here to assist with technical issues or
                booking questions.
              </p>
            </div>
            <div className="p-8 bg-neutral-soft rounded-3xl border border-gray-100 text-left cursor-pointer hover:bg-neutral-soft/80 transition-colors">
              <ShieldCheck className="w-8 h-8 text-medic-primary mb-4" />
              <h4 className="font-bold text-gray-900 text-lg mb-2">
                {" "}
                HIPAA Compliant
              </h4>
              <p className="text-gray-500 text-sm font-medium">
                We adhere to the highest standards of data privacy and security
                protections.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Services;

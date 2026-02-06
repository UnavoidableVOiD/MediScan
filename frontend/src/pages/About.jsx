import React from 'react';
import { motion } from 'framer-motion';
import { HeartPulse, ShieldCheck, Cpu, Heart, Users, Globe, Target, Zap } from 'lucide-react';

const About = () => {
    const sections = [
        {
            icon: Target,
            title: "Purpose & Vision",
            content: "Mediscan was born from a simple yet powerful idea: making complex medical data understandable for everyone. We envision a world where patients are empowered with clear, AI-driven insights into their health, and doctors have efficient tools to support their diagnostics.",
            color: "text-blue-600",
            bg: "bg-blue-50"
        },
        {
            icon: Cpu,
            title: "AI in Healthcare",
            content: "Our AI engine is trained to identify and explain key health indicators from medical reports. By automating the extraction and summary of data, we reduce the cognitive load for both patients and providers, allowing for more meaningful medical consultations.",
            color: "text-medic-dark",
            bg: "bg-medic-light/20"
        },
        {
            icon: ShieldCheck,
            title: "Ethical & Responsible AI",
            content: "Privacy and fairness are at the core of Mediscan. We use state-of-the-art encryption (HIPAA-aligned) and ensure our AI explanations are informational, transparent, and always remind users to consult with qualified medical professionals.",
            color: "text-medic-accent",
            bg: "bg-medic-accent/10"
        }
    ];

    return (
        <div className="min-h-screen bg-white">
            {/* Hero Section */}
            <section className="relative py-24 overflow-hidden bg-medic-dark text-white">
                <div className="max-w-7xl mx-auto px-6 relative z-10">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="max-w-3xl"
                    >
                        <h1 className="text-5xl md:text-6xl font-bold mb-6 tracking-tight">
                            Empowering healthcare with AI insights.
                        </h1>
                        <p className="text-xl text-medic-light/80 leading-relaxed max-w-2xl">
                            Mediscan is a next-generation medical analysis platform designed to bridge the gap between raw data and actionable health knowledge.
                        </p>
                    </motion.div>
                </div>
                {/* Abstract Illustration Background */}
                <div className="absolute right-0 top-0 w-1/2 h-full opacity-10 pointer-events-none hidden lg:block">
                    <svg viewBox="0 0 400 400" className="w-full h-full">
                        <path d="M50,100 C150,50 250,150 350,100 L350,300 C250,350 150,250 50,300 Z" fill="white" />
                    </svg>
                </div>
            </section>

            {/* Project Overview */}
            <section className="py-24 max-w-7xl mx-auto px-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
                    <div>
                        <h2 className="text-3xl font-bold text-gray-900 mb-6 flex items-center gap-3">
                            <span className="w-1.5 h-8 bg-medic-dark rounded-full"></span>
                            What is Mediscan?
                        </h2>
                        <div className="space-y-6 text-lg text-gray-600 leading-relaxed">
                            <p>
                                Mediscan is an intelligent health assistant that simplifies medical reports. We believe that medical knowledge shouldn't be locked behind complex terminology and dense data tables.
                            </p>
                            <p>
                                Our platform uses advanced Optical Character Recognition (OCR) and Natural Language Processing (NLP) to read your lab results and generate easy-to-understand summaries. Whether you're a patient tracking chronic health or a doctor needing a quick summary, Mediscan is built for you.
                            </p>
                        </div>
                    </div>
                    <div className="grid grid-cols-2 gap-6">
                        <div className="bg-neutral-soft p-8 rounded-3xl space-y-4">
                            <div className="bg-white w-12 h-12 rounded-2xl flex items-center justify-center shadow-sm">
                                <Zap className="w-6 h-6 text-medic-dark" />
                            </div>
                            <h4 className="font-bold text-xl">Fast & Efficient</h4>
                        </div>
                        <div className="bg-medic-light/10 p-8 rounded-3xl space-y-4 mt-8">
                            <div className="bg-white w-12 h-12 rounded-2xl flex items-center justify-center shadow-sm">
                                <Globe className="w-6 h-6 text-medic-accent" />
                            </div>
                            <h4 className="font-bold text-xl">Global Access</h4>
                        </div>
                    </div>
                </div>
            </section>

            {/* Core Values / Specific Sections */}
            <section className="bg-neutral-soft/50 py-24">
                <div className="max-w-7xl mx-auto px-6">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                        {sections.map((item, index) => (
                            <motion.div
                                key={index}
                                initial={{ opacity: 0, y: 30 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: index * 0.1 }}
                                className="bg-white p-10 rounded-[2.5rem] shadow-xl shadow-medic-dark/5 border border-white hover:border-medic-light transition-colors"
                            >
                                <div className={`w-14 h-14 ${item.bg} rounded-2xl flex items-center justify-center mb-6`}>
                                    <item.icon className={`w-7 h-7 ${item.color}`} />
                                </div>
                                <h3 className="text-2xl font-bold text-gray-900 mb-4">{item.title}</h3>
                                <p className="text-gray-500 leading-relaxed">
                                    {item.content}
                                </p>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Final CTA/Statement */}
            <section className="py-24 max-w-5xl mx-auto px-6 text-center">
                <div className="bg-medic-dark rounded-[3rem] p-16 text-white overflow-hidden relative">
                    <Heart className="absolute -top-10 -right-10 w-64 h-64 text-white/5" />
                    <h2 className="text-4xl font-bold mb-6">Our commitment is to your health.</h2>
                    <p className="text-medic-light/70 text-lg mb-0 max-w-2xl mx-auto">
                        We are continuously refining our models and interfaces to ensure the highest degree of accuracy and empathy in every medical insight we provide.
                    </p>
                </div>
            </section>
        </div>
    );
};

export default About;

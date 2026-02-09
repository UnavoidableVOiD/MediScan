import React from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, Upload, Shield, Zap, FileText, CheckCircle } from 'lucide-react';
import { Link } from 'react-router-dom';
import { toast } from 'react-toastify';

const LandingPage = () => {
    const handleDemoClick = () => {
        toast.info("Entering Demo Mode — No login required.", {
            icon: <CheckCircle className="text-medic-dark" />
        });
    };

    return (
        <div className="flex flex-col">
            {/* Hero Section */}
            <section className="relative overflow-hidden pt-20 pb-32 bg-gradient-to-b from-medic-light/30 to-white">
                <div className="max-w-7xl mx-auto px-6 grid md:grid-cols-2 gap-12 items-center">
                    <motion.div
                        initial={{ opacity: 0, x: -50 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.8 }}
                    >
                        <h1 className="text-5xl md:text-6xl font-bold text-medic-dark leading-tight mb-6">
                            Understand Your <br />
                            <span className="text-medic-accent">Medical Reports</span> <br />
                            Instantly.
                        </h1>
                        <p className="text-lg text-gray-600 mb-10 max-w-lg leading-relaxed">
                            Mediscan uses advanced AI to decode complex medical terminology,
                            giving you clear, actionable insights from your laboratory results
                            in seconds.
                        </p>
                        <div className="flex flex-wrap gap-4 items-center">
                            <Link to="/login" className="bg-medic-dark text-white px-8 py-3.5 rounded-full font-bold shadow-lg shadow-medic-dark/20 hover:bg-medic-primary transition-all hover:-translate-y-1">
                                Login
                            </Link>
                            <Link to="/signup" className="bg-white text-medic-dark border-2 border-medic-dark/10 px-8 py-3.5 rounded-full font-bold hover:bg-medic-light transition-all hover:-translate-y-1">
                                Sign Up
                            </Link>
                            <Link
                                to="/demo"
                                onClick={handleDemoClick}
                                className="flex items-center gap-1 group text-medic-dark font-bold ml-2 hover:underline decoration-medic-dark/30 underline-offset-8"
                            >
                                Try Demo <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
                            </Link>
                        </div>
                    </motion.div>

                    <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.8, delay: 0.2 }}
                        className="relative"
                    >
                        <div className="absolute -inset-4 bg-medic-dark/5 rounded-3xl blur-2xl -z-10"></div>
                        <div className="bg-white p-8 rounded-3xl shadow-xl shadow-medic-dark/5 border border-medic-light/50">
                            {/* Abstract Medical Graphic Representation */}
                            <div className="w-full aspect-square bg-medic-light/20 rounded-2xl flex items-center justify-center border-2 border-dashed border-medic-dark/20 relative overflow-hidden">
                                <motion.div
                                    animate={{ y: [0, -10, 0] }}
                                    transition={{ repeat: Infinity, duration: 4, ease: "easeInOut" }}
                                    className="flex flex-col items-center gap-4 text-medic-dark"
                                >
                                    <FileText className="w-24 h-24 opacity-80" />
                                    <div className="flex flex-col gap-2 w-48">
                                        <div className="h-2 w-full bg-medic-dark/20 rounded-full overflow-hidden">
                                            <motion.div
                                                animate={{ x: ['-100%', '100%'] }}
                                                transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                                                className="h-full w-20 bg-medic-accent"
                                            />
                                        </div>
                                        <div className="h-2 w-2/3 bg-medic-dark/20 rounded-full" />
                                        <div className="h-2 w-1/2 bg-medic-dark/20 rounded-full" />
                                    </div>
                                </motion.div>
                                {/* Decorative blobs */}
                                <div className="absolute top-4 right-4 w-12 h-12 bg-medic-accent/10 rounded-full blur-xl animate-pulse"></div>
                                <div className="absolute bottom-10 left-10 w-20 h-20 bg-medic-dark/5 rounded-full blur-2xl"></div>
                            </div>
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* Features Section */}
            <section className="py-24 bg-white">
                <div className="max-w-7xl mx-auto px-6">
                    <div className="text-center mb-20">
                        <h2 className="text-3xl md:text-4xl font-bold text-medic-dark mb-4">Why choose Mediscan?</h2>
                        <div className="h-1.5 w-20 bg-medic-accent mx-auto rounded-full opacity-30"></div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
                        {[
                            { icon: Upload, title: "Upload Reports", desc: "Easily upload PDFs or scans of your medical documents." },
                            { icon: Zap, title: "AI Analysis", desc: "Our models interpret data points with medical precision." },
                            { icon: Shield, title: "Secure & Private", desc: "Your data is encrypted and handled with strict confidentiality." },
                            { icon: CheckCircle, title: "Fast Results", desc: "Get comprehensive summaries in less than 5 seconds." }
                        ].map((f, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, y: 20 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: i * 0.1 }}
                                whileHover={{ y: -5 }}
                                className="p-8 rounded-2xl border border-gray-100 hover:border-medic-light hover:shadow-lg hover:shadow-medic-dark/5 transition-all text-center group"
                            >
                                <div className="w-14 h-14 bg-medic-light/30 rounded-xl flex items-center justify-center mx-auto mb-6 group-hover:bg-medic-dark group-hover:text-white transition-colors">
                                    <f.icon className="w-7 h-7 text-medic-dark group-hover:text-white transition-colors" />
                                </div>
                                <h3 className="text-xl font-bold text-gray-900 mb-3">{f.title}</h3>
                                <p className="text-gray-600 text-sm leading-relaxed">{f.desc}</p>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Demo Section */}
            <section className="py-24 bg-neutral-background">
                <div className="max-w-7xl mx-auto px-6">
                    <div className="bg-white rounded-[2rem] overflow-hidden shadow-2xl shadow-medic-dark/10 border border-medic-light/50 flex flex-col md:flex-row">
                        <div className="md:w-1/2 p-12 flex flex-col justify-center">
                            <span className="text-medic-accent font-bold uppercase tracking-widest text-xs mb-4">Sample Analysis</span>
                            <h2 className="text-3xl md:text-4xl font-bold text-medic-dark mb-6">See Mediscan in Action.</h2>
                            <p className="text-gray-600 mb-8 leading-relaxed">
                                Take a look at how our AI transforms complex laboratory data into
                                clear, readable sections you can easily understand.
                            </p>
                            <div className="flex flex-col gap-4">
                                <div className="flex items-center gap-3 text-sm text-gray-700">
                                    <CheckCircle className="w-5 h-5 text-medic-accent" />
                                    <span>No login required for trial</span>
                                </div>
                                <div className="flex items-center gap-3 text-sm text-gray-700">
                                    <CheckCircle className="w-5 h-5 text-medic-accent" />
                                    <span>Protected healthcare data</span>
                                </div>
                            </div>
                            <Link to="/demo" className="mt-10 bg-medic-dark text-white text-center px-10 py-4 rounded-full font-bold hover:bg-medic-primary transition-all w-fit shadow-md shadow-medic-dark/10">
                                View Demo Report
                            </Link>
                        </div>

                        <div className="md:w-1/2 bg-medic-light/10 p-12 border-l border-medic-light/20 flex flex-col gap-6 justify-center">
                            <div className="flex items-center justify-between px-4 py-3 bg-white/60 rounded-xl border border-medic-light/30 text-xs">
                                <span className="font-bold text-gray-400">INPUT REPORT</span>
                                <FileText className="w-4 h-4 text-gray-300" />
                            </div>

                            <div className="relative">
                                <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 mb-8 blur-[1px]">
                                    <div className="h-4 w-3/4 bg-gray-100 rounded mb-3" />
                                    <div className="h-3 w-1/2 bg-gray-50 rounded mb-6" />
                                    <div className="flex justify-between items-center pb-2 border-b border-gray-50 mb-2">
                                        <div className="h-3 w-20 bg-gray-100 rounded" />
                                        <div className="h-3 w-12 bg-gray-100 rounded" />
                                    </div>
                                    <div className="flex justify-between items-center pb-2 border-b border-gray-50">
                                        <div className="h-3 w-20 bg-gray-100 rounded" />
                                        <div className="h-3 w-12 bg-gray-100 rounded" />
                                    </div>
                                </div>

                                <ArrowRight className="absolute left-1/2 -bottom-4 translate-x-[-50%] w-8 h-8 text-medic-accent bg-white rounded-full p-2 shadow-md border border-medic-light/20 z-10 rotate-90" />

                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    whileInView={{ opacity: 1, y: 0 }}
                                    viewport={{ once: true }}
                                    className="bg-medic-dark p-6 rounded-2xl shadow-xl mt-4"
                                >
                                    <div className="flex items-center gap-2 mb-4">
                                        <div className="w-2 h-2 rounded-full bg-medic-accent animate-pulse" />
                                        <span className="text-[10px] font-bold text-medic-light/60 uppercase tracking-widest">AI Result</span>
                                    </div>
                                    <h4 className="text-white font-bold mb-2">Optimal Health Summary</h4>
                                    <p className="text-medic-light/80 text-xs leading-relaxed">
                                        Your blood glucose levels (92 mg/dL) are within the normal fasting range.
                                        Lipid panel shows slightly elevated LDL cholesterol...
                                    </p>
                                </motion.div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    );
};

export default LandingPage;

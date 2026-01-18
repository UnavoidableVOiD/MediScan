import React from 'react';
import { Link } from 'react-router-dom';
import { Shield, Activity, Lock, Zap, ArrowRight, FileText } from 'lucide-react';

const LandingPage = () => {
    return (
        <div className="pt-20">
            {/* Hero Section */}
            <section className="relative overflow-hidden bg-white py-16 lg:py-24">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
                    <div className="flex flex-col lg:flex-row items-center gap-16">
                        <div className="flex-1 text-center lg:text-left">
                            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-50 text-blue-700 font-bold text-sm mb-6 border border-blue-100">
                                <span className="relative flex h-3 w-3">
                                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                                    <span className="relative inline-flex rounded-full h-3 w-3 bg-blue-500"></span>
                                </span>
                                Trusted by 10,000+ Users
                            </div>
                            <h1 className="text-5xl lg:text-7xl font-extrabold text-gray-900 leading-[1.1] mb-8">
                                Understand Your <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-emerald-500">Health Reports</span> Like Never Before
                            </h1>
                            <p className="text-xl text-gray-600 mb-10 leading-relaxed max-w-2xl">
                                MediScan uses advanced AI to decode complex medical jargon into clear, actionable insights. Upload your reports and get instant clarity.
                            </p>
                            <div className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-4">
                                <Link
                                    to="/signup"
                                    className="w-full sm:w-auto px-10 py-5 bg-gradient-to-r from-blue-600 to-emerald-500 text-white rounded-2xl font-bold text-lg shadow-xl shadow-blue-200 hover:shadow-2xl hover:-translate-y-1 transition-all flex items-center justify-center gap-2"
                                >
                                    Get Started Free
                                    <ArrowRight className="h-5 w-5" />
                                </Link>
                                <Link
                                    to="/demo"
                                    className="w-full sm:w-auto px-10 py-5 bg-white text-gray-700 border-2 border-gray-100 rounded-2xl font-bold text-lg hover:border-blue-100 hover:bg-blue-50/50 transition-all flex items-center justify-center gap-2"
                                >
                                    Try Live Demo
                                </Link>
                            </div>
                        </div>
                        <div className="flex-1 relative">
                            <div className="relative z-10 animate-float">
                                <div className="bg-white p-8 rounded-3xl shadow-2xl shadow-blue-100 border border-gray-100">
                                    <div className="flex items-center gap-4 mb-6">
                                        <div className="h-12 w-12 bg-emerald-100 rounded-xl flex items-center justify-center">
                                            <Activity className="h-6 w-6 text-emerald-600" />
                                        </div>
                                        <div>
                                            <h4 className="font-bold text-gray-900">Health Summary</h4>
                                            <p className="text-sm text-gray-500">AI Analysis Complete</p>
                                        </div>
                                    </div>
                                    <div className="space-y-4">
                                        <div className="h-3 w-full bg-gray-100 rounded-full overflow-hidden text-xs">
                                            <div className="h-full w-[85%] bg-blue-500 rounded-full"></div>
                                        </div>
                                        <div className="h-3 w-[70%] bg-gray-100 rounded-full overflow-hidden text-xs">
                                            <div className="h-full w-full bg-emerald-400 rounded-full"></div>
                                        </div>
                                        <div className="h-3 w-[90%] bg-gray-100 rounded-full overflow-hidden text-xs">
                                            <div className="h-full w-full bg-blue-400 rounded-full"></div>
                                        </div>
                                    </div>
                                    <div className="mt-8 pt-6 border-t border-gray-50 flex justify-between items-center text-sm font-bold">
                                        <span className="text-gray-500 uppercase tracking-wider text-xs">Accuracy Rate</span>
                                        <span className="text-emerald-600">99.8% Secured</span>
                                    </div>
                                </div>
                            </div>
                            {/* Decorative Blobs */}
                            <div className="absolute -top-20 -right-20 h-64 w-64 bg-blue-100 rounded-full blur-3xl opacity-50"></div>
                            <div className="absolute -bottom-20 -left-20 h-64 w-64 bg-emerald-100 rounded-full blur-3xl opacity-50"></div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Stats Section */}
            <section className="bg-gray-50 py-16 border-y border-gray-100">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                        <div className="text-center">
                            <div className="text-4xl font-extrabold text-blue-600 mb-2">10M+</div>
                            <div className="text-gray-500 font-medium font-bold">Reports Analyzed</div>
                        </div>
                        <div className="text-center">
                            <div className="text-4xl font-extrabold text-emerald-500 mb-2">99.8%</div>
                            <div className="text-gray-500 font-medium font-bold">AI Accuracy</div>
                        </div>
                        <div className="text-center">
                            <div className="text-4xl font-extrabold text-blue-600 mb-2">24/7</div>
                            <div className="text-gray-500 font-medium font-bold">Instant Access</div>
                        </div>
                        <div className="text-center">
                            <div className="text-4xl font-extrabold text-emerald-500 mb-2">100%</div>
                            <div className="text-gray-500 font-medium font-bold">Data Secured</div>
                        </div>
                    </div>
                </div>
            </section>

            {/* How it Works Section */}
            <section className="py-24 bg-white overflow-hidden">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="text-center mb-20">
                        <h2 className="text-4xl font-bold text-gray-900 mb-4">How MediScan Works</h2>
                        <p className="text-xl text-gray-500 max-w-2xl mx-auto leading-relaxed">Get your analysis in three simple steps.</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-12 relative">
                        <div className="relative text-center group">
                            <div className="w-20 h-20 bg-blue-50 text-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-8 transition-transform group-hover:scale-110 group-hover:bg-blue-600 group-hover:text-white group-hover:rotate-3 shadow-lg shadow-blue-50 duration-300">
                                <FileText className="h-10 w-10" />
                            </div>
                            <h3 className="text-2xl font-bold text-gray-900 mb-4">1. Upload Report</h3>
                            <p className="text-gray-600 leading-relaxed font-medium">Simply upload your PDF or image medical report to our secure platform.</p>
                            {/* Connector line (Desktop only) */}
                            <div className="hidden md:block absolute top-10 left-[60%] w-full h-[2px] bg-gradient-to-r from-blue-100 to-emerald-100 -z-10"></div>
                        </div>

                        <div className="relative text-center group">
                            <div className="w-20 h-20 bg-emerald-50 text-emerald-600 rounded-2xl flex items-center justify-center mx-auto mb-8 transition-transform group-hover:scale-110 group-hover:bg-emerald-500 group-hover:text-white group-hover:-rotate-3 shadow-lg shadow-emerald-50 duration-300">
                                <Zap className="h-10 w-10" />
                            </div>
                            <h3 className="text-2xl font-bold text-gray-900 mb-4">2. AI Processing</h3>
                            <p className="text-gray-600 leading-relaxed font-medium">Our clinical-grade AI analyzes every metric and medical terminology.</p>
                            {/* Connector line (Desktop only) */}
                            <div className="hidden md:block absolute top-10 left-[60%] w-full h-[2px] bg-gradient-to-r from-emerald-100 to-blue-100 -z-10"></div>
                        </div>

                        <div className="relative text-center group">
                            <div className="w-20 h-20 bg-blue-50 text-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-8 transition-transform group-hover:scale-110 group-hover:bg-blue-600 group-hover:text-white group-hover:rotate-3 shadow-lg shadow-blue-50 duration-300">
                                <Shield className="h-10 w-10" />
                            </div>
                            <h3 className="text-2xl font-bold text-gray-900 mb-4">3. Get Insights</h3>
                            <p className="text-gray-600 leading-relaxed font-medium">Receive a clear, easy-to-understand breakdown of your health data.</p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Enhanced Features Grid */}
            <section id="features" className="py-24 bg-gray-50">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="text-center mb-16">
                        <h2 className="text-4xl font-bold text-gray-900 mb-4 font-extrabold">Comprehensive Health Analysis</h2>
                        <p className="text-xl text-gray-500 font-medium">Everything you need to understand your health journey.</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                        {[
                            {
                                icon: <Zap className="h-6 w-6" />,
                                title: "Instant Breakdown",
                                desc: "Translate complex medical terms into simple, everyday language instantly.",
                                color: "bg-blue-50 text-blue-600"
                            },
                            {
                                icon: <Activity className="h-6 w-6" />,
                                title: "Trend Analysis",
                                desc: "Compare multiple reports to track how your health metrics change over time.",
                                color: "bg-emerald-50 text-emerald-600"
                            },
                            {
                                icon: <Lock className="h-6 w-6" />,
                                title: "Bank-Level Security",
                                desc: "Your data is encrypted end-to-end and stored with strict privacy standards.",
                                color: "bg-blue-50 text-blue-600"
                            },
                            {
                                icon: <Shield className="h-6 w-6" />,
                                title: "Medical Accuracy",
                                desc: "AI models trained on millions of clinical records for high-precision results.",
                                color: "bg-emerald-50 text-emerald-600"
                            },
                            {
                                icon: <FileText className="h-6 w-6" />,
                                title: "PDF Extraction",
                                desc: "No manual data entry. Just upload your file and we'll extract everything automatically.",
                                color: "bg-blue-50 text-blue-600"
                            },
                            {
                                icon: <ArrowRight className="h-6 w-6" />,
                                title: "Smart Suggestions",
                                desc: "Get helpful notes on what to discuss with your doctor based on results.",
                                color: "bg-emerald-50 text-emerald-600"
                            }
                        ].map((feature, idx) => (
                            <div key={idx} className="bg-white p-10 rounded-3xl border border-gray-100 shadow-sm hover:shadow-xl hover:-translate-y-1 transition-all duration-300">
                                <div className={`w-14 h-14 ${feature.color} rounded-2xl flex items-center justify-center mb-8`}>
                                    {feature.icon}
                                </div>
                                <h3 className="text-2xl font-bold text-gray-900 mb-4">{feature.title}</h3>
                                <p className="text-gray-600 leading-relaxed font-medium">{feature.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* FAQ Section */}
            <section className="py-24 bg-white border-t border-gray-50">
                <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
                    <h2 className="text-4xl font-extrabold text-gray-900 mb-12 text-center">Frequently Asked Questions</h2>
                    <div className="space-y-6">
                        {[
                            { q: "Is my medical data safe?", a: "Yes, we use military-grade encryption and comply with global health data privacy standards. Your data is yours alone." },
                            { q: "Is this a substitute for a doctor?", a: "Not at all. MediScan is a tool to help you understand your reports so you can have better, more informed conversations with your doctor." },
                            { q: "What types of reports can I upload?", a: "We support most laboratory tests, blood work, radiology reports, and general clinical summaries in PDF or Image format." }
                        ].map((item, idx) => (
                            <div key={idx} className="bg-gray-50 p-8 rounded-2xl border border-gray-100 group hover:border-blue-200 transition-colors">
                                <h4 className="text-xl font-bold text-gray-900 mb-2">{item.q}</h4>
                                <p className="text-gray-600 leading-relaxed font-medium">{item.a}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Final CTA Section */}
            <section className="py-24 px-4 sm:px-6 lg:px-8">
                <div className="max-w-7xl mx-auto bg-gradient-to-br from-blue-700 to-emerald-600 rounded-3xl p-12 lg:p-24 text-center relative overflow-hidden shadow-2xl shadow-blue-200">
                    <div className="relative z-10">
                        <h2 className="text-4xl lg:text-6xl font-extrabold text-white mb-8 leading-tight">Take Control of Your Health Data</h2>
                        <p className="text-xl text-blue-50 mb-12 max-w-2xl mx-auto opcity-90 font-medium">
                            Join thousands of others who are simplifying their medical journeys with MediScan. Start your first analysis today.
                        </p>
                        <div className="flex flex-col sm:flex-row justify-center gap-6">
                            <Link
                                to="/signup"
                                className="px-12 py-5 bg-white text-blue-700 font-bold text-xl rounded-2xl hover:bg-blue-50 transition-all shadow-xl"
                            >
                                Get Started Free
                            </Link>
                            <Link
                                to="/demo"
                                className="px-12 py-5 bg-blue-600 text-white border-2 border-blue-500 font-bold text-xl rounded-2xl hover:bg-blue-700 transition-all"
                            >
                                Try the Demo
                            </Link>
                        </div>
                    </div>
                    {/* Background decoration */}
                    <div className="absolute top-0 right-0 -translate-y-1/2 translate-x-1/2 w-[500px] h-[500px] bg-white opacity-10 rounded-full blur-[100px]"></div>
                    <div className="absolute bottom-0 left-0 translate-y-1/2 -translate-x-1/2 w-[500px] h-[500px] bg-emerald-400 opacity-20 rounded-full blur-[100px]"></div>
                </div>
            </section>

            {/* Footer */}
            <footer className="bg-white border-t border-gray-100 py-16">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex flex-col md:flex-row justify-between items-center gap-12 text-center md:text-left">
                        <div className="flex flex-col items-center md:items-start gap-4">
                            <div className="flex items-center gap-2">
                                <div className="bg-blue-600 p-1.5 rounded-lg">
                                    <Shield className="h-6 w-6 text-white" />
                                </div>
                                <span className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-700 to-emerald-600">MediScan</span>
                            </div>
                            <p className="text-gray-500 text-sm max-w-xs font-medium">
                                Empowering patients with AI-driven health insights and simplified medical reports.
                            </p>
                        </div>
                        <div className="grid grid-cols-2 lg:grid-cols-3 gap-12 sm:gap-16">
                            <div>
                                <h4 className="font-bold text-gray-900 mb-6 uppercase text-sm tracking-widest">Platform</h4>
                                <ul className="space-y-4 text-gray-500 font-bold">
                                    <li><a href="#features" className="hover:text-blue-600 transition-colors">Features</a></li>
                                    <li><Link to="/demo" className="hover:text-blue-600 transition-colors">Demo</Link></li>
                                    <li><Link to="/signup" className="hover:text-blue-600 transition-colors">Get Started</Link></li>
                                </ul>
                            </div>
                            <div>
                                <h4 className="font-bold text-gray-900 mb-6 uppercase text-sm tracking-widest">Support</h4>
                                <ul className="space-y-4 text-gray-500 font-bold">
                                    <li><a href="#" className="hover:text-blue-600 transition-colors">Help Center</a></li>
                                    <li><a href="#" className="hover:text-blue-600 transition-colors">Contact Us</a></li>
                                    <li><a href="#" className="hover:text-blue-600 transition-colors">API Docs</a></li>
                                </ul>
                            </div>
                            <div className="col-span-2 lg:col-span-1">
                                <h4 className="font-bold text-gray-900 mb-6 uppercase text-sm tracking-widest text-center lg:text-left">Legal</h4>
                                <ul className="space-y-4 text-gray-500 font-bold text-center lg:text-left">
                                    <li><a href="#" className="hover:text-blue-600 transition-colors">Privacy Policy</a></li>
                                    <li><a href="#" className="hover:text-blue-600 transition-colors">Terms of Service</a></li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    <div className="mt-16 pt-8 border-t border-gray-50 text-center text-gray-400 text-sm font-bold">
                        © {new Date().getFullYear()} MediScan. Built for healthier futures.
                    </div>
                </div>
            </footer>
        </div>
    );
};

export default LandingPage;

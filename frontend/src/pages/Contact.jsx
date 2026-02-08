import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Send, Mail, Phone, MapPin, MessageCircle, Loader2 } from 'lucide-react';
import { toast } from 'react-toastify';

const Contact = () => {
    const [loading, setLoading] = useState(false);
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        message: ''
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
            setFormData({ name: '', email: '', message: '' });
        }, 1500);
    };

    return (
        <div className="min-h-screen bg-neutral-background py-20 px-6">
            <div className="max-w-6xl mx-auto">
                {/* Header */}
                <div className="text-center mb-20">
                    <motion.h1
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="text-4xl md:text-5xl font-bold text-medic-dark mb-4 tracking-tight"
                    >
                        Contact Us
                    </motion.h1>
                    <p className="text-gray-500 text-lg max-w-xl mx-auto">
                        We're here to help — reach out anytime with questions, feedback, or support requests.
                    </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-start">
                    {/* Contact Form */}
                    <motion.div
                        initial={{ opacity: 0, x: -30 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="lg:col-span-12 xl:col-span-7 bg-white p-8 md:p-12 rounded-[2.5rem] shadow-xl shadow-medic-dark/5"
                    >
                        <form onSubmit={handleSubmit} className="space-y-6">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div className="space-y-1.5">
                                    <label className="text-xs font-bold text-gray-400 uppercase tracking-widest block ml-1">Your Name</label>
                                    <input
                                        type="text"
                                        name="name"
                                        required
                                        value={formData.name}
                                        onChange={handleChange}
                                        placeholder="John Doe"
                                        className="w-full px-6 py-4 bg-neutral-soft rounded-2xl border-transparent focus:border-medic-dark focus:bg-white transition-all outline-none text-sm font-medium"
                                    />
                                </div>
                                <div className="space-y-1.5">
                                    <label className="text-xs font-bold text-gray-400 uppercase tracking-widest block ml-1">Email Address</label>
                                    <input
                                        type="email"
                                        name="email"
                                        required
                                        value={formData.email}
                                        onChange={handleChange}
                                        placeholder="john@example.com"
                                        className="w-full px-6 py-4 bg-neutral-soft rounded-2xl border-transparent focus:border-medic-dark focus:bg-white transition-all outline-none text-sm font-medium"
                                    />
                                </div>
                            </div>
                            <div className="space-y-1.5">
                                <label className="text-xs font-bold text-gray-400 uppercase tracking-widest block ml-1">Your Message</label>
                                <textarea
                                    name="message"
                                    required
                                    rows="6"
                                    value={formData.message}
                                    onChange={handleChange}
                                    placeholder="How can we help you?"
                                    className="w-full px-6 py-4 bg-neutral-soft rounded-2xl border-transparent focus:border-medic-dark focus:bg-white transition-all outline-none text-sm font-medium resize-none"
                                ></textarea>
                            </div>

                            <button
                                type="submit"
                                disabled={loading}
                                className="w-full md:w-auto px-12 py-4 bg-medic-dark text-white rounded-2xl font-bold shadow-lg shadow-medic-dark/20 hover:bg-medic-primary transition-all active:scale-95 disabled:opacity-50 flex items-center justify-center gap-2"
                            >
                                {loading ? (
                                    <>
                                        <Loader2 className="w-5 h-5 animate-spin" />
                                        Sending...
                                    </>
                                ) : (
                                    <>
                                        <Send className="w-5 h-5" />
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
                        <div className="bg-medic-dark p-10 rounded-[2.5rem] text-white overflow-hidden relative">
                            <MessageCircle className="absolute -bottom-10 -right-10 w-48 h-48 opacity-10" />
                            <h3 className="text-2xl font-bold mb-8 relative z-10">Direct Support</h3>

                            <div className="space-y-6 relative z-10">
                                <div className="flex items-start gap-5">
                                    <div className="w-12 h-12 bg-white/10 rounded-2xl flex items-center justify-center flex-shrink-0">
                                        <Mail className="w-6 h-6 text-medic-light" />
                                    </div>
                                    <div>
                                        <p className="text-sm font-bold text-medic-light/50 uppercase tracking-widest mb-1">Email Us</p>
                                        <p className="text-lg font-medium">support@mediscan.com</p>
                                    </div>
                                </div>

                                <div className="flex items-start gap-5">
                                    <div className="w-12 h-12 bg-white/10 rounded-2xl flex items-center justify-center flex-shrink-0">
                                        <Phone className="w-6 h-6 text-medic-light" />
                                    </div>
                                    <div>
                                        <p className="text-sm font-bold text-medic-light/50 uppercase tracking-widest mb-1">Call Us</p>
                                        <p className="text-lg font-medium">+1 (555) 765-4321</p>
                                    </div>
                                </div>

                                <div className="flex items-start gap-5">
                                    <div className="w-12 h-12 bg-white/10 rounded-2xl flex items-center justify-center flex-shrink-0">
                                        <MapPin className="w-6 h-6 text-medic-light" />
                                    </div>
                                    <div>
                                        <p className="text-sm font-bold text-medic-light/50 uppercase tracking-widest mb-1">Address</p>
                                        <p className="text-lg font-medium whitespace-pre-line">
                                            123 Digital Health Way, Suite 400{"\n"}
                                            San Francisco, CA 94103
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="bg-white p-10 rounded-[2.5rem] border border-medic-light/20 shadow-xl shadow-medic-dark/5">
                            <h4 className="font-bold text-gray-900 mb-4 tracking-tight">Need a quick answer?</h4>
                            <p className="text-gray-500 mb-6">
                                Check our FAQ page for common questions about report analysis, account security, and data privacy.
                            </p>
                            <a href="/faq" className="text-medic-dark font-bold hover:underline">Read FAQ &rarr;</a>
                        </div>
                    </motion.div>
                </div>
            </div>
        </div>
    );
};

export default Contact;

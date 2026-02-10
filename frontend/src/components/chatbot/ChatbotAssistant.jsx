import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare, X, Send, HeartPulse, User, Bot, Info } from 'lucide-react';
import { useSelector } from 'react-redux';

const ChatbotAssistant = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        { role: 'bot', content: 'Hello! I am your Mediscan Assistant. How can I help you understand your reports today?' }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const { user, isAuthenticated } = useSelector(state => state.auth);

    // Show for patients (case-insensitive check)
    if (!isAuthenticated || user?.role?.toLowerCase() !== 'patient') return null;

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMsg = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsLoading(true);

        const botMsgId = Date.now();
        setMessages(prev => [...prev, { role: 'bot', content: '', id: botMsgId }]);

        try {
            const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000/api'}/chatbot/chat/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include',
                body: JSON.stringify({
                    question: input,
                    patient_context: "The user is viewing their medical dashboard."
                }),
            });

            if (!response.ok) throw new Error('Failed to connect to AI service');

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let done = false;

            while (!done) {
                const { value, done: doneReading } = await reader.read();
                done = doneReading;
                const chunkValue = decoder.decode(value, { stream: true });

                setMessages(prev => prev.map(msg =>
                    msg.id === botMsgId
                        ? { ...msg, content: msg.content + chunkValue }
                        : msg
                ));
            }
        } catch (err) {
            console.error(err);
            setMessages(prev => prev.map(msg =>
                msg.id === botMsgId
                    ? { ...msg, content: "I'm sorry, I'm having trouble connecting to the AI service right now. Please try again later." }
                    : msg
            ));
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <>
            <button
                onClick={() => setIsOpen(true)}
                className="fixed bottom-8 right-8 w-16 h-16 bg-medic-dark text-white rounded-full shadow-2xl shadow-medic-dark/30 flex items-center justify-center hover:bg-medic-primary transition-all active:scale-90 z-40 group"
            >
                <MessageSquare className="w-7 h-7" />
                <span className="absolute -top-1 -right-1 w-4 h-4 bg-medic-accent border-2 border-white rounded-full animate-ping"></span>
            </button>

            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9, y: 50, x: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0, x: 0 }}
                        exit={{ opacity: 0, scale: 0.9, y: 50, x: 20 }}
                        className="fixed bottom-8 right-8 w-[400px] h-[600px] bg-white rounded-[2.5rem] shadow-2xl border border-medic-light/50 flex flex-col overflow-hidden z-50 origin-bottom-right"
                    >
                        {/* Header */}
                        <div className="bg-medic-dark p-6 text-white flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center">
                                    <HeartPulse className="w-6 h-6 text-medic-accent" />
                                </div>
                                <div>
                                    <h3 className="font-bold">AI Assistant</h3>
                                    <div className="flex items-center gap-1.5 text-[10px] text-medic-light/60 font-bold uppercase tracking-widest">
                                        <div className="w-1.5 h-1.5 rounded-full bg-medic-accent animate-pulse" />
                                        Online Always
                                    </div>
                                </div>
                            </div>
                            <button
                                onClick={() => setIsOpen(false)}
                                className="w-10 h-10 hover:bg-white/10 rounded-full flex items-center justify-center transition-colors"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        {/* Messages */}
                        <div className="flex-grow overflow-y-auto p-6 space-y-6 bg-neutral-background/30">
                            {messages.map((msg, i) => (
                                <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                    <div className={`flex gap-3 max-w-[85%] ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${msg.role === 'user' ? 'bg-medic-dark text-white' : 'bg-medic-light text-medic-dark'}`}>
                                            {msg.role === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
                                        </div>
                                        <div className={`p-4 rounded-2xl text-sm leading-relaxed shadow-sm ${msg.role === 'user' ? 'bg-medic-dark text-white rounded-tr-none' : 'bg-white text-gray-700 rounded-tl-none border border-gray-100'}`}>
                                            {msg.content}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Disclaimer & Input */}
                        <div className="p-6 border-t border-gray-100 bg-white space-y-4">
                            <div className="flex gap-2 p-3 bg-amber-50 rounded-xl border border-amber-100">
                                <Info className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
                                <p className="text-[10px] text-amber-700 leading-tight">
                                    This assistant provides informational guidance only and is not a medical diagnosis.
                                </p>
                            </div>

                            <form onSubmit={handleSend} className="relative">
                                <input
                                    type="text"
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    placeholder="Ask about your report..."
                                    className="w-full pl-5 pr-14 py-3.5 bg-neutral-soft border-transparent focus:bg-white focus:border-medic-dark/20 rounded-2xl text-sm outline-none transition-all"
                                />
                                <button
                                    type="submit"
                                    className="absolute right-2 top-2 w-10 h-10 bg-medic-dark text-white rounded-xl flex items-center justify-center hover:bg-medic-primary transition-all active:scale-95"
                                >
                                    <Send className="w-5 h-5" />
                                </button>
                            </form>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
};

export default ChatbotAssistant;

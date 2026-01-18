import React, { useState } from 'react';
import { useLocation } from 'react-router-dom';
import { MessageSquare, Send, X, Bot, User, AlertCircle, Maximize2, Minimize2 } from 'lucide-react';

const Chatbot = () => {
    const location = useLocation();
    const [isOpen, setIsOpen] = useState(false);
    const [isMinimized, setIsMinimized] = useState(false);

    const [message, setMessage] = useState('');
    const [chat, setChat] = useState([
        { role: 'bot', text: 'Hello! I am your MediScan assistant. How can I help you understand your reports today?' }
    ]);

    // Hidden on Profile page
    if (location.pathname === '/profile') return null;

    const handleSend = (e) => {
        e.preventDefault();
        if (!message.trim()) return;

        setChat([...chat, { role: 'user', text: message }]);
        setMessage('');

        // Mock bot response
        setTimeout(() => {
            setChat(prev => [...prev, { role: 'bot', text: "I'm a demo assistant. In the full version, I'll analyze your specific report data to provide detailed explanations." }]);
        }, 1000);
    };

    if (!isOpen) {
        return (
            <button
                onClick={() => setIsOpen(true)}
                className="fixed bottom-6 right-6 bg-gradient-to-r from-blue-600 to-emerald-500 text-white p-4 rounded-full shadow-2xl shadow-blue-200 hover:scale-110 hover:-translate-y-1 transition-all z-50 group"
            >
                <div className="relative">
                    <MessageSquare className="h-7 w-7" />
                    <span className="absolute -top-1 -right-1 flex h-3 w-3">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                    </span>
                </div>
                {/* Tooltip */}
                <div className="absolute right-full mr-4 top-1/2 -translate-y-1/2 px-3 py-1 bg-gray-900 text-white text-xs font-bold rounded-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                    Ask MediScan AI
                </div>
            </button>
        );
    }

    return (
        <div className={`fixed bottom-6 right-6 w-96 max-w-[calc(100vw-3rem)] bg-white rounded-3xl shadow-2xl border border-gray-100 z-50 flex flex-col transition-all duration-300 ${isMinimized ? 'h-16' : 'h-[550px]'}`}>
            {/* Header */}
            <div className="p-4 bg-gradient-to-r from-blue-600 to-emerald-500 rounded-t-[22px] flex items-center justify-between text-white">
                <div className="flex items-center gap-3">
                    <div className="bg-white/20 p-2 rounded-xl backdrop-blur-sm">
                        <Bot className="h-5 w-5" />
                    </div>
                    <div>
                        <h3 className="font-bold text-sm">MediScan Assistant</h3>
                        {!isMinimized && <p className="text-[10px] text-blue-100 font-medium leading-none">AI Health Partner</p>}
                    </div>
                </div>
                <div className="flex items-center gap-1">
                    <button onClick={() => setIsMinimized(!isMinimized)} className="p-1.5 hover:bg-white/10 rounded-lg transition-colors">
                        {isMinimized ? <Maximize2 className="h-4 w-4" /> : <Minimize2 className="h-4 w-4" />}
                    </button>
                    <button onClick={() => setIsOpen(false)} className="p-1.5 hover:bg-white/10 rounded-lg transition-colors">
                        <X className="h-4 w-4" />
                    </button>
                </div>
            </div>

            {!isMinimized && (
                <>
                    {/* Disclaimer */}
                    <div className="px-4 py-2 bg-amber-50 border-b border-amber-100 flex items-start gap-2">
                        <AlertCircle className="h-4 w-4 text-amber-500 mt-0.5 flex-shrink-0" />
                        <p className="text-[10px] text-amber-700 font-bold leading-tight uppercase tracking-wider">
                            Disclaimer: Information only. Not a medical diagnosis. Consult a doctor.
                        </p>
                    </div>

                    {/* Chat Area */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-gray-200">
                        {chat.map((msg, i) => (
                            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                <div className={`flex gap-2 max-w-[80%] ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                                    <div className={`flex-shrink-0 h-8 w-8 rounded-xl flex items-center justify-center ${msg.role === 'user' ? 'bg-blue-600' : 'bg-gray-100'}`}>
                                        {msg.role === 'user' ? <User className="h-4 w-4 text-white" /> : <Bot className="h-4 w-4 text-gray-500" />}
                                    </div>
                                    <div className={`p-3 rounded-2xl text-sm font-medium ${msg.role === 'user'
                                        ? 'bg-blue-600 text-white rounded-tr-none'
                                        : 'bg-gray-100 text-gray-800 rounded-tl-none'
                                        }`}>
                                        {msg.text}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Input Area */}
                    <form onSubmit={handleSend} className="p-4 border-t border-gray-100 bg-gray-50/50 rounded-b-3xl">
                        <div className="relative flex items-center gap-2">
                            <input
                                type="text"
                                value={message}
                                onChange={(e) => setMessage(e.target.value)}
                                placeholder="Type a message..."
                                className="flex-1 bg-white border border-gray-200 rounded-xl px-4 py-2.5 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all placeholder:text-gray-400"
                            />
                            <button
                                type="submit"
                                className="bg-blue-600 text-white p-2.5 rounded-xl hover:bg-blue-700 transition-all shadow-md shadow-blue-100"
                            >
                                <Send className="h-5 w-5" />
                            </button>
                        </div>
                    </form>
                </>
            )}
        </div>
    );
};

export default Chatbot;

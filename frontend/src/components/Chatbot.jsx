import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext.jsx';

export default function Chatbot() {
    const { user } = useAuth();
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        {
            text: "Hello! I'm MediBot. I can help explain your medical reports or answer general health questions.",
            sender: 'bot'
        }
    ]);
    const [input, setInput] = useState('');
    const [showPopup, setShowPopup] = useState(false);

    React.useEffect(() => {
        // Show popup after 3 seconds
        const timer = setTimeout(() => {
            setShowPopup(true);
        }, 3000);
        return () => clearTimeout(timer);
    }, []);

    if (!user || user.role !== 'patient') return null;

    const isPremium = user.plan === 'premium';

    const handleSend = () => {
        if (!input.trim()) return;

        if (!isPremium && messages.length > 4) {
            setMessages((prev) => [
                ...prev,
                { text: input, sender: 'user' },
                { text: '🔒 Upgrade to Premium to continue chatting with MediBot.', sender: 'bot' }
            ]);
            setInput('');
            return;
        }

        const newMsgs = [...messages, { text: input, sender: 'user' }];
        setMessages(newMsgs);
        const prompt = input;
        setInput('');

        setTimeout(() => {
            let response = 'I’m not sure about that. Please consult a doctor.';
            const lower = prompt.toLowerCase();
            if (lower.includes('anemia') || lower.includes('hemoglobin'))
                response =
                    'Low hemoglobin typically indicates anemia. Try iron-rich foods (spinach, legumes, red meat) and consult your doctor for iron studies.';
            else if (lower.includes('diabetes') || lower.includes('sugar'))
                response =
                    'High glucose can suggest pre-diabetes. Regular exercise, weight management, and an HbA1c test are commonly advised.';
            else if (lower.includes('hi') || lower.includes('hello')) response = 'Hello! How can I help you today?';

            setMessages((prev) => [...prev, { text: response, sender: 'bot' }]);
        }, 900);
    };

    return (
        <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end">
            {isOpen && (
                <div className="bg-white w-80 h-96 rounded-2xl shadow-2xl border border-slate-200 mb-4 flex flex-col overflow-hidden fade-in">
                    <div
                        className={`${isPremium ? 'bg-gradient-to-r from-blue-600 to-indigo-600' : 'bg-slate-700'
                            } p-4 text-white flex justify-between items-center`}
                    >
                        <div className="flex items-center gap-2">
                            <i className="fa-solid fa-robot"></i>
                            <span className="font-semibold">
                                MediBot{' '}
                            </span>
                        </div>
                        <button onClick={() => setIsOpen(false)} className="hover:bg-white/20 p-1 rounded">
                            <i className="fa-solid fa-xmark"></i>
                        </button>
                    </div>

                    <div className="flex-1 p-4 overflow-y-auto bg-slate-50 space-y-3 text-sm flex flex-col">
                        {messages.map((m, i) => (
                            <div
                                key={i}
                                className={`p-3 max-w-[85%] ${m.sender === 'user'
                                    ? 'bg-blue-600 text-white self-end rounded-l-xl rounded-tr-xl ml-auto'
                                    : 'bg-white border border-slate-200 text-slate-800 self-start rounded-r-xl rounded-tl-xl shadow-sm'
                                    }`}
                            >
                                {m.text}
                            </div>
                        ))}

                        {!isPremium && messages.length > 4 && (
                            <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-center">
                                <p className="text-xs text-yellow-800 mb-2">You've reached the free message limit.</p>
                                <Link
                                    to="/membership"
                                    className="block w-full bg-yellow-500 text-white text-xs font-bold py-1.5 rounded hover:bg-yellow-600"
                                >
                                    Unlock Unlimited Chat
                                </Link>
                            </div>
                        )}
                    </div>

                    <div className="p-3 border-t border-slate-200 bg-white flex gap-2">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                            placeholder="Ask a question..."
                            className="flex-1 border border-slate-300 rounded-full px-4 py-2 text-sm focus:outline-none focus:border-blue-500"
                        />
                        <button
                            onClick={handleSend}
                            className="bg-blue-600 text-white rounded-full w-9 h-9 flex items-center justify-center hover:bg-blue-700 transition"
                        >
                            <i className="fa-solid fa-paper-plane text-xs"></i>
                        </button>
                    </div>
                </div>
            )}

            {!isOpen && showPopup && (
                <div className="bg-white p-4 rounded-2xl shadow-xl border border-slate-200 mb-4 mr-2 animate-bounce-in relative select-none flex items-center gap-3 max-w-[240px]">
                    <div className="bg-gradient-to-br from-blue-500 to-indigo-600 w-10 h-10 rounded-full flex items-center justify-center text-white shadow-md shrink-0">
                        <i className="fa-solid fa-robot text-lg"></i>
                    </div>
                    <div>
                        <p className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-0.5">MediBot</p>
                        <p className="text-sm font-bold text-slate-800 leading-tight">Hi! I'm your AI assistant.</p>
                    </div>

                    <div className="absolute -bottom-2 right-8 w-4 h-4 bg-white border-b border-r border-slate-200 transform rotate-45"></div>

                    <button
                        onClick={(e) => { e.stopPropagation(); setShowPopup(false); }}
                        className="absolute -top-2 -right-2 bg-slate-100 rounded-full p-1 w-6 h-6 flex items-center justify-center hover:bg-slate-200 text-xs text-slate-500 shadow-sm border border-slate-200"
                    >
                        <i className="fa-solid fa-xmark"></i>
                    </button>
                </div>
            )}

            <button
                onClick={() => setIsOpen(!isOpen)}
                className={`${isPremium ? 'bg-linear-to-r from-blue-600 to-indigo-600' : 'bg-slate-700'
                    } hover:opacity-90 text-white rounded-full w-14 h-14 shadow-lg flex items-center justify-center transition-transform hover:scale-105`}
            >
                <i className={`fa-solid ${isOpen ? 'fa-xmark' : 'fa-message'} text-2xl`}></i>
            </button>
        </div>
    );
}
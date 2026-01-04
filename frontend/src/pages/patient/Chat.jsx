import React, { useState, useRef, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Send, Bot, User, Loader2 } from 'lucide-react';

const Chat = () => {
    const [messages, setMessages] = useState([
        { id: 1, role: 'bot', text: 'Hello! I am MediBot AI. How can I help you regarding your health reports today?' }
    ]);
    const [input, setInput] = useState('');
    const [typing, setTyping] = useState(false);
    const scrollRef = useRef(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages, typing]);

    const handleSend = () => {
        if (!input.trim()) return;

        const userMsg = { id: Date.now(), role: 'user', text: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setTyping(true);

        // Simulate AI delay
        setTimeout(() => {
            const botMsg = {
                id: Date.now() + 1,
                role: 'bot',
                text: "I analyzed your latest values. Your cholesterol levels are slightly elevated (210 mg/dL). I recommend reducing saturated fats and increasing fiber intake."
            };
            setMessages(prev => [...prev, botMsg]);
            setTyping(false);
        }, 2000);
    };

    return (
        <div className="h-[calc(100vh-8rem)] flex flex-col">
            <div className="mb-4">
                <h1 className="text-3xl font-bold tracking-tight text-primary">Chat with MediBot</h1>
                <p className="text-muted-foreground">Ask questions about your reports, medicines, or symptoms.</p>
            </div>

            <Card className="flex-1 flex flex-col overflow-hidden border-2">
                {/* Chat Area */}
                <div className="flex-1 p-4 overflow-y-auto space-y-4 bg-muted/10">
                    {messages.map((msg) => (
                        <div
                            key={msg.id}
                            className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                            <div className={`flex gap-3 max-w-[80%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                                <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${msg.role === 'user' ? 'bg-primary text-primary-foreground' : 'bg-green-600 text-white'
                                    }`}>
                                    {msg.role === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
                                </div>
                                <div className={`p-3 rounded-lg text-sm ${msg.role === 'user'
                                        ? 'bg-primary text-primary-foreground rounded-tr-none'
                                        : 'bg-card border shadow-sm rounded-tl-none'
                                    }`}>
                                    {msg.text}
                                </div>
                            </div>
                        </div>
                    ))}

                    {typing && (
                        <div className="flex w-full justify-start">
                            <div className="flex gap-3 max-w-[80%]">
                                <div className="w-8 h-8 rounded-full bg-green-600 text-white flex items-center justify-center shrink-0">
                                    <Bot className="w-5 h-5" />
                                </div>
                                <div className="p-3 rounded-lg bg-card border shadow-sm rounded-tl-none flex items-center gap-1">
                                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></span>
                                </div>
                            </div>
                        </div>
                    )}
                    <div ref={scrollRef} />
                </div>

                {/* Input Area */}
                <div className="p-4 bg-card border-t flex gap-2">
                    <Input
                        placeholder="Type a message..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                        className="flex-1"
                    />
                    <Button onClick={handleSend} size="icon" disabled={typing}>
                        <Send className="w-4 h-4" />
                    </Button>
                </div>
            </Card>
        </div>
    );
};

export default Chat;

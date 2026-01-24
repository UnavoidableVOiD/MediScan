import React, { useState, useRef, useEffect } from 'react';

function Chat() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const messagesEndRef = useRef(null);
  const [sending, setSending] = useState(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || sending) return;

    const newMessage = {
      text: inputMessage,
      sender: 'patient',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages([...messages, newMessage]);
    setInputMessage('');
    setSending(true);

    // Simulate doctor response
    setTimeout(() => {
      setMessages(prev => [...prev, {
        text: 'Thank you for your message. Our medical team will respond shortly.',
        sender: 'doctor',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }]);
      setSending(false);
    }, 1000);
  };

  const doctors = [
    { id: 1, name: 'Dr. Sarah Johnson', specialization: 'Cardiologist', status: 'online', avatar: null },
    { id: 2, name: 'Dr. Michael Chen', specialization: 'General Physician', status: 'online', avatar: null },
    { id: 3, name: 'Dr. Emily Davis', specialization: 'Pediatrician', status: 'offline', avatar: null }
  ];

  return (
    <div className="flex-1 min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 px-4 py-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-slate-900 mb-2">Chat with Doctors</h1>
          <p className="text-slate-600 text-lg">Connect with healthcare professionals for consultations</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Doctors List */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-lg border border-slate-200 p-4">
              <h2 className="text-lg font-bold text-slate-900 mb-4">Available Doctors</h2>
              <div className="space-y-2">
                {doctors.map((doctor) => (
                  <div
                    key={doctor.id}
                    onClick={() => setSelectedDoctor(doctor)}
                    className={`p-3 rounded-lg cursor-pointer transition ${
                      selectedDoctor?.id === doctor.id
                        ? 'bg-green-50 border-2 border-green-600'
                        : 'bg-slate-50 hover:bg-slate-100 border-2 border-transparent'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div className="relative">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-green-500 to-green-600 flex items-center justify-center text-white font-semibold">
                          {doctor.name.split(' ').map(n => n[0]).join('')}
                        </div>
                        <span
                          className={`absolute bottom-0 right-0 w-3 h-3 rounded-full border-2 border-white ${
                            doctor.status === 'online' ? 'bg-green-500' : 'bg-slate-400'
                          }`}
                        ></span>
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-semibold text-slate-900 text-sm truncate">{doctor.name}</p>
                        <p className="text-xs text-slate-600 truncate">{doctor.specialization}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Chat Window */}
          <div className="lg:col-span-3">
            <div className="bg-white rounded-xl shadow-lg border border-slate-200 h-[600px] flex flex-col">
              {/* Chat Header */}
              {selectedDoctor ? (
                <div className="p-4 border-b border-slate-200 bg-gradient-to-r from-green-50 to-white">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-green-500 to-green-600 flex items-center justify-center text-white font-semibold">
                      {selectedDoctor.name.split(' ').map(n => n[0]).join('')}
                    </div>
                    <div>
                      <h3 className="font-bold text-slate-900">{selectedDoctor.name}</h3>
                      <p className="text-xs text-slate-600">{selectedDoctor.specialization}</p>
                    </div>
                    <span className={`ml-auto px-3 py-1 text-xs font-semibold rounded-full ${
                      selectedDoctor.status === 'online' ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-600'
                    }`}>
                      {selectedDoctor.status === 'online' ? 'Online' : 'Offline'}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="p-4 border-b border-slate-200">
                  <p className="text-slate-500 text-center">Select a doctor to start chatting</p>
                </div>
              )}

              {/* Messages Area */}
              <div className="flex-1 p-6 overflow-y-auto space-y-4 bg-slate-50">
                {!selectedDoctor ? (
                  <div className="text-center text-slate-500 mt-20">
                    <i className="fa-solid fa-comments text-5xl mb-4 opacity-50"></i>
                    <p className="text-lg font-semibold mb-2">Select a doctor to start chatting</p>
                    <p className="text-sm">Choose from the list on the left to begin a conversation</p>
                  </div>
                ) : messages.length === 0 ? (
                  <div className="text-center text-slate-500 mt-20">
                    <i className="fa-solid fa-inbox text-5xl mb-4 opacity-50"></i>
                    <p className="text-lg font-semibold mb-2">No messages yet</p>
                    <p className="text-sm">Start a conversation with {selectedDoctor.name}</p>
                  </div>
                ) : (
                  messages.map((message, index) => (
                    <div
                      key={index}
                      className={`flex ${message.sender === 'patient' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div className={`flex flex-col max-w-xs lg:max-w-md ${message.sender === 'patient' ? 'items-end' : 'items-start'}`}>
                        <div
                          className={`px-4 py-3 rounded-2xl ${
                            message.sender === 'patient'
                              ? 'bg-gradient-to-r from-green-600 to-green-500 text-white rounded-br-sm'
                              : 'bg-white text-slate-900 border border-slate-200 rounded-bl-sm'
                          }`}
                        >
                          <p className="text-sm">{message.text}</p>
                        </div>
                        <span className="text-xs text-slate-500 mt-1 px-2">{message.timestamp}</span>
                      </div>
                    </div>
                  ))
                )}
                {sending && (
                  <div className="flex justify-start">
                    <div className="bg-white px-4 py-3 rounded-2xl rounded-bl-sm border border-slate-200">
                      <i className="fa-solid fa-ellipsis animate-pulse text-slate-400"></i>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              {selectedDoctor && (
                <form onSubmit={handleSendMessage} className="p-4 border-t border-slate-200 bg-white rounded-b-xl">
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={inputMessage}
                      onChange={(e) => setInputMessage(e.target.value)}
                      placeholder="Type your message..."
                      className="flex-1 px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-600 focus:border-transparent"
                      disabled={sending}
                    />
                    <button
                      type="submit"
                      disabled={sending || !inputMessage.trim()}
                      className="bg-green-600 text-white font-semibold px-6 py-3 rounded-lg hover:bg-green-700 transition disabled:bg-slate-400 disabled:cursor-not-allowed shadow-lg hover:shadow-xl"
                    >
                      <i className="fa-solid fa-paper-plane"></i>
                    </button>
                  </div>
                </form>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Chat;

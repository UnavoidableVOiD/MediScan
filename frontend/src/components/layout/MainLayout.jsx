import React from 'react';
import { Outlet } from 'react-router-dom';
import Navbar from './Navbar';
import Footer from './Footer';
import ChatbotAssistant from '../chatbot/ChatbotAssistant';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const MainLayout = () => {
    return (
        <div className="min-h-screen flex flex-col bg-neutral-background text-gray-900 font-body">
            <Navbar />
            <main className="flex-grow">
                <Outlet />
            </main>
            <ChatbotAssistant />
            <Footer />
            <ToastContainer
                position="bottom-right"
                autoClose={3000}
                hideProgressBar={false}
                newestOnTop
                closeOnClick
                rtl={false}
                pauseOnFocusLoss
                draggable
                pauseOnHover
                theme="light"
            />
        </div>
    );
};

export default MainLayout;

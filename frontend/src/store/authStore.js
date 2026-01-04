import { create } from 'zustand';

export const useAuthStore = create((set) => ({
    user: null, // { name: 'John Doe', role: 'patient' | 'doctor' | 'admin', email: '...', id: '...' }
    isAuthenticated: false,

    login: (userData) => set({ user: userData, isAuthenticated: true }),
    logout: () => set({ user: null, isAuthenticated: false }),

    // Mock login function for demo
    mockLogin: (role) => {
        const mockUsers = {
            patient: { id: 'p1', name: 'Alex Carter', email: 'alex@example.com', role: 'patient' },
            doctor: { id: 'd1', name: 'Dr. Sarah Smith', email: 'sarah@mediscan.com', role: 'doctor' },
            admin: { id: 'a1', name: 'Admin User', email: 'admin@mediscan.com', role: 'admin' },
        };

        // Simulate API delay
        return new Promise((resolve) => {
            setTimeout(() => {
                set({ user: mockUsers[role], isAuthenticated: true });
                resolve(mockUsers[role]);
            }, 800);
        });
    }
}));

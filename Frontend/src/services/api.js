import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const api = axios.create({
    baseURL: API_URL,
    withCredentials: true,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const authApi = {
    // Corrected to match doctor/urls.py
    submitDoctorVerification: (formData) => api.put('/doctor/verify/', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    }),
    updateProfile: (data) => api.patch('/auth/profile/', data),
};

export const doctorApi = {
    getStats: () => api.get('/doctor/my-patients/stats/'),
    getPatients: () => api.get('/doctor/my-patients/'),
    getPatientReports: (id) => api.get(`/doctor/my-patients/${id}/reports/`),
    updatePatientNotes: (id, notes) => api.post(`/doctor/my-patients/${id}/update_notes/`, { notes }),
    addComment: (data) => api.post('/doctor/comments/', data),
};

export const appointmentApi = {
    getAvailability: (doctorId) => api.get(`/doctor/availability/?doctor=${doctorId}`),
    manageAvailability: (data) => api.post('/doctor/availability/', data),
    syncAvailability: (data) => api.post('/doctor/availability/sync/', data),
    deleteAvailability: (id) => api.delete(`/doctor/availability/${id}/`),
    getAppointments: () => api.get('/doctor/appointments/'),
    bookAppointment: (data) => api.post('/doctor/appointments/', data),
    verifyPayment: (appointmentId, data) => api.post(`/doctor/appointments/${appointmentId}/verify_payment/`, data),
    getRecommendedDoctors: (specialization) => api.get(`/doctor/list/?specialization=${specialization}`),
};

// Response interceptor to handle token refresh
api.interceptors.response.use(
    (response) => response,
    async (error) => {
        const originalRequest = error.config;

        // If error is 401 and we haven't tried to refresh yet
        if (error.response?.status === 401 && !originalRequest._retry) {
            originalRequest._retry = true;

            try {
                // Attempt to refresh the token
                // The refresh cookie is HttpOnly, so the backend will pick it up
                await axios.post(`${API_URL}/auth/token/refresh/`, {}, { withCredentials: true });

                // If refresh succeeds, retry the original request
                return api(originalRequest);
            } catch (refreshError) {
                // If refresh fails, redirect to login or handle as needed
                // For now, let the error bubble up to be handled by the slice
                return Promise.reject(refreshError);
            }
        }

        return Promise.reject(error);
    }
);


export const adminApi = {
    login: (credentials) => api.post('/admin/login/', credentials),
    getDoctors: (status) => api.get(`/admin/doctors/${status ? `?status=${status}` : ''}`),
    verifyDoctor: (id, data) => api.patch(`/admin/verify-doctor/${id}/`, data),
    getPatients: () => api.get('/admin/patients/'),
    createAdmin: (data) => api.post('/admin/create-admin/', data),
    unverifyDoctor: (id) => api.post(`/admin/doctors/${id}/unverify/`),
    // Flexible methods for user management
    updateDoctor: (id, data) => api.patch(`/admin/doctors/${id}/`, data),
    deleteDoctor: (id) => api.delete(`/admin/doctors/${id}/`),
    updatePatient: (id, data) => api.patch(`/admin/patients/${id}/`, data),
    deletePatient: (id) => api.delete(`/admin/patients/${id}/`),
};

export default api;


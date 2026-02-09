import axios from 'axios';

const api = axios.create({
    baseURL: 'http://localhost:8000/api',
    withCredentials: true,
    headers: {
        'Content-Type': 'application/json',
    },
});

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
                await axios.post('http://localhost:8000/api/auth/token/refresh/', {}, { withCredentials: true });

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

export default api;


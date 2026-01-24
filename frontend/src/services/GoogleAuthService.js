export const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;

/**
 * Loads the Google Identity Services script.
 * @returns {Promise<void>}
 */
export const loadGoogleScript = () => {
    return new Promise((resolve, reject) => {
        if (window.google) {
            resolve();
            return;
        }
        const script = document.createElement('script');
        script.src = 'https://accounts.google.com/gsi/client';
        script.async = true;
        script.defer = true;
        script.onload = () => resolve();
        script.onerror = (err) => reject(err);
        document.body.appendChild(script);
    });
};

/**
 * Initializes Google Sign-In with the given callback.
 * @param {Function} callback - The callback function to handle the Google response.
 */
export const initializeGoogleSignIn = (callback) => {
    if (!GOOGLE_CLIENT_ID) {
        console.error('GOOGLE_CLIENT_ID is missing. Please add VITE_GOOGLE_CLIENT_ID to your .env file.');
        alert('Google Authentication Error: Missing Client ID.\n\nPlease add your VITE_GOOGLE_CLIENT_ID to the frontend .env file.');
        return;
    }
    if (window.google?.accounts?.id) {
        window.google.accounts.id.initialize({
            client_id: GOOGLE_CLIENT_ID,
            callback: callback,
        });
    } else {
        console.error('Google Identity Services not loaded');
        alert('Google Identity Services failed to load. Please check your internet connection or console for details.');
    }
};

/**
 * Renders the Google Sign-In button in the specified element.
 * @param {string} elementId - The ID of the element to render the button in.
 * @param {Object} options - Customization options for the button.
 */
export const renderGoogleButton = (elementId, options = {}) => {
    const defaultOptions = {
        theme: 'outline',
        size: 'large',
        text: 'continue_with',
        shape: 'rectangular',
        ...options
    };

    const element = document.getElementById(elementId);
    if (element && window.google?.accounts?.id) {
        window.google.accounts.id.renderButton(element, defaultOptions);
    }
};

/**
 * Decodes the JWT credential from Google.
 * @param {string} credential - The JWT credential.
 * @returns {Object} - The decoded user information.
 */
export const decodeGoogleCredential = (credential) => {
    try {
        const base64Url = credential.split('.')[1];
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const jsonPayload = decodeURIComponent(atob(base64).split('').map(c => {
            return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }).join(''));

        return JSON.parse(jsonPayload);
    } catch (error) {
        console.error('Error decoding Google credential:', error);
        return null;
    }
};

import { API_URL } from './env';

/**
 * Helper function to fetch data from the API
 * @param {string} endpoint - The API endpoint to call
 * @param {Object} options - Fetch options
 * @returns {Promise<any>} The API response as JSON
 */
export const fetchApi = async (endpoint, options = {}) => {
  try {
    const response = await fetch(`${API_URL || ''}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      }
    });

    if (!response.ok) {
      throw new Error(`API request failed with status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`API request error for ${endpoint}:`, error);
    throw error;
  }
};

/**
 * Get player configuration settings from the server
 * @returns {Promise<{auto_stop_playback: boolean, player_controls: boolean, progress_bar: boolean, loading_indicator: boolean}>}
 */
export const getPlayerConfig = async () => {
  try {
    return await fetchApi('/config/player');
  } catch (error) {
    console.warn('Failed to load player config, using defaults:', error);
    return {
      auto_stop_playback: true,
      player_controls: true,
      progress_bar: true,
      loading_indicator: true
    };
  }
};
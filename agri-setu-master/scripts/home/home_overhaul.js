/**
 * home_overhaul.js
 * Logic for the new 'Digital Agrarian' home page.
 * Handles weather updates and top recommendation preview.
 */

const API_BASE = window.location.origin;

// -- Weather Logic --
const weatherApiKey = '1cb42169e693e7b28f8cf8a4bc2ae3ff';

function getWeatherIcon(weatherMain) {
    switch (weatherMain.toLowerCase()) {
        case 'clear': return 'sunny';
        case 'clouds': return 'cloudy';
        case 'rain': return 'rainy';
        case 'drizzle': return 'rainy';
        case 'thunderstorm': return 'thunderstorm';
        case 'snow': return 'snowy';
        case 'mist':
        case 'fog': return 'foggy';
        default: return 'sunny';
    }
}

async function updateWeather(city = 'Kochi') {
    const url = `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(city)}&appid=${weatherApiKey}&units=metric`;
    
    try {
        const res = await fetch(url);
        if (!res.ok) throw new Error('Weather fetch failed');
        const data = await res.json();
        
        document.getElementById('weather-city').textContent = `${data.name}, ${data.sys.country}`;
        document.getElementById('weather-temp').textContent = `${Math.round(data.main.temp)}°C`;
        document.getElementById('weather-desc').textContent = data.weather[0].description.charAt(0).toUpperCase() + data.weather[0].description.slice(1);
        document.getElementById('weather-humidity').textContent = `${data.main.humidity}%`;
        document.getElementById('weather-wind').textContent = `${Math.round(data.wind.speed * 3.6)} km/h`;
        
        const iconEl = document.getElementById('weather-icon');
        iconEl.textContent = getWeatherIcon(data.weather[0].main);
        if (data.weather[0].main.toLowerCase() === 'clear') {
            iconEl.style.fontVariationSettings = "'FILL' 1";
        } else {
            iconEl.style.fontVariationSettings = "'FILL' 0";
        }
    } catch (err) {
        console.error('Weather error:', err);
    }
}

// -- Top Recommendation Logic --
async function updateTopRecommendation() {
    try {
        const res = await fetch(`${API_BASE}/api/top-recommendation`);
        const data = await res.json();
        
        if (data.success && data.top) {
            const top = data.top;
            document.getElementById('top-crop-name').innerHTML = top.crop_name;
            document.getElementById('top-crop-desc').textContent = top.recommendation_text || `Optimal conditions detected for ${top.crop_name}. High ROI potential based on current market trends.`;
            document.getElementById('top-yield-prob').textContent = `${(top.hybrid_score * 100).toFixed(0)}%`;
            document.getElementById('top-market-price').textContent = `Rs. ${top.predicted_price.toLocaleString('en-IN')}`;
            document.getElementById('top-harvest-cycle').innerHTML = `${top.approx_months * 30} <span class="text-lg">Days</span>`;
            
            // Map crop image if available locally, else use default or keep existing
            const imgEl = document.getElementById('top-crop-image');
            const imgName = top.crop_name.toLowerCase() === 'rice' ? 'paddy' : top.crop_name.toLowerCase();
            const localImg = `images/${imgName}.jpg`;
            // We can check if image exists or just use a fallback
            imgEl.src = localImg; 
            imgEl.onerror = () => { imgEl.src = 'https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&q=80&w=2000'; };
        }
    } catch (err) {
        console.error('Recommendation error:', err);
    }
}

// -- Initialize --
document.addEventListener('DOMContentLoaded', () => {
    updateWeather();
    updateTopRecommendation();
});

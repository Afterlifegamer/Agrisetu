// Replace 'YOUR_API_KEY' with your actual data
const apiKey = '1cb42169e693e7b28f8cf8a4bc2ae3ff';

function getWeatherIcon(weatherMain) {
  switch (weatherMain.toLowerCase()) {
    case 'clear':
      return 'sunny';
    case 'clouds':
      return 'cloudy';
    case 'rain':
      return 'rainy';
    case 'drizzle':
      return 'rainy';
    case 'thunderstorm':
      return 'thunderstorm';
    case 'snow':
      return 'snowy';
    case 'mist':
    case 'fog':
      return 'foggy';
    default:
      return 'sunny';
  }
}

export function weatherCard(city)
{
const url = `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(city)}&appid=${apiKey}&units=metric`; // Use units=metric for Celsius

fetch(url)
  .then(response => {
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return response.json();
  })
  .then(data => {
    let currentWeather = `${data.main.temp}°C, ${data.weather[0].description}`;
    console.log(`Current temperature in ${data.name}: ${currentWeather}`);
    // Update the DOM
    const weatherDisplay = document.getElementById('weather-display');
    if (weatherDisplay) {
      weatherDisplay.textContent = currentWeather;
    }
    const weatherIcon = document.getElementById('weather-icon');
    if (weatherIcon) {
      weatherIcon.textContent = getWeatherIcon(data.weather[0].main);
    }
  })
  .catch(error => {
    console.error('There has been a problem with your fetch operation:', error);
    const weatherDisplay = document.getElementById('weather-display');
    if (weatherDisplay) {
      weatherDisplay.textContent = '';
    }
    const weatherIcon = document.getElementById('weather-icon');
    if (weatherIcon) {
      weatherIcon.textContent = 'sunny';
    }
  });
}
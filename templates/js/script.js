// script.js
const PEXELS_API_KEY = 'HpLa0UJwJdnSIGZ8hvefPlttzgPTOqoTkMkwihwECufrmo5z4oYFmZ4S'; // Replace with your Pexels API key

async function fetchSriLankanImages(query = 'Sri Lanka attractions') {
  try {
    const response = await fetch(`https://api.pexels.com/v1/search?query=${query}&per_page=10`, {
      headers: {
        Authorization: PEXELS_API_KEY,
      },
    });
    const data = await response.json();
    return data.photos.map(photo => photo.src.medium);
  } catch (error) {
    console.error('Error fetching images:', error);
    return ['https://via.placeholder.com/300x200?text=Sri+Lanka'];
  }
}

function animateElement(element) {
  element.style.opacity = '0';
  element.style.transform = 'translateY(20px)';
  setTimeout(() => {
    element.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    element.style.opacity = '1';
    element.style.transform = 'translateY(0)';
  }, 100);
}
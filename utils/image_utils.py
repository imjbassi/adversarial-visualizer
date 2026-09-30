import urllib.parse
from io import BytesIO

import requests
from PIL import Image

USER_AGENT = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
              'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36')
REQUEST_TIMEOUT = 15


def search_pexels(search_term, api_key):
    """Return an image URL for the search term via the Pexels API, or None."""
    if not api_key:
        return None
    try:
        url = ('https://api.pexels.com/v1/search?query='
               f'{urllib.parse.quote(search_term)}&per_page=1')
        response = requests.get(
            url,
            headers={'User-Agent': USER_AGENT, 'Authorization': api_key},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 200:
            photos = response.json().get('photos', [])
            if photos:
                return photos[0]['src']['medium']
    except requests.RequestException:
        pass
    return None


def get_placeholder_url(search_term):
    """A deterministic random photo URL (not term-relevant), used as fallback."""
    seed = urllib.parse.quote(search_term)
    return f'https://picsum.photos/seed/{seed}/400/400'


def find_image_url(search_term, pexels_api_key=None):
    """Find an image URL for a search term, falling back to a placeholder."""
    url = search_pexels(search_term, pexels_api_key)
    return url if url else get_placeholder_url(search_term)


def load_image_from_url(url):
    """Download an image and return it as an RGB PIL Image."""
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
    }
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    content_type = response.headers.get('Content-Type', '')
    if content_type and not content_type.startswith('image/'):
        raise ValueError(f'URL returned non-image content: {content_type}')

    return Image.open(BytesIO(response.content)).convert('RGB')


def load_image_from_file(path):
    """Open a local image file as an RGB PIL Image."""
    return Image.open(path).convert('RGB')

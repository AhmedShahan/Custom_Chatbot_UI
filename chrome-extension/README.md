# RAG Chatbot Chrome Extension

This Chrome extension adds a chat bubble to any web page, which opens the RAG Chatbot in an iframe when clicked.

## Installation

1. Download or clone this repository
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode" by toggling the switch in the top-right corner
4. Click "Load unpacked" and select the `chrome-extension` folder
5. The extension is now installed and should appear in your Chrome toolbar

## Usage

1. Click the RAG Chatbot icon in your Chrome toolbar to toggle the chat bubble on the current page
2. Click the blue chat bubble that appears in the bottom-right corner of the page
3. Enter the URL where your RAG Chatbot is running (e.g., `http://127.0.0.1:8501/`)
4. The chatbot will open in an iframe on the page
5. Use the chat interface to interact with the RAG Chatbot
6. Click the X button to close the chatbot

**Note:** On HTTPS sites (like Google or Facebook), you might see an error when trying to load an HTTP chatbot URL due to browser security restrictions. If this happens, you can use the "Open in New Tab" button in the error message.

## Development

The extension consists of the following files:

- `manifest.json`: Extension configuration
- `popup.html`: HTML for the popup that appears when clicking the extension icon
- `popup.js`: JavaScript for the popup
- `content.js`: JavaScript that injects the chat bubble into web pages
- `styles.css`: Styling for the chat bubble and window
- `background.js`: Background service worker for the extension
- `icon.svg`: Extension icon

## Permissions

The extension requires the following permissions:
- `activeTab`: To access the current tab
- `scripting`: To inject scripts into web pages
- `http://localhost:*/`: To access local development servers

## Troubleshooting

If the chat bubble doesn't appear:
1. Make sure the extension is enabled
2. Try clicking the extension icon in the toolbar again
3. Refresh the page and try again

If the chatbot doesn't load in the iframe:
1. Make sure the URL you entered is correct
2. Ensure your RAG Chatbot server is running at the specified URL
3. Check for console errors by opening Chrome DevTools (F12)

## Handling 403 (Forbidden) Errors

If you encounter a 403 Forbidden error or see the "Error Loading Chatbot" message, this is likely due to security restrictions. Here are some ways to fix it:

### If you're running Streamlit locally

When running your Streamlit app, use these flags:
```
streamlit run app.py --server.enableCORS=false --server.enableXsrfProtection=false --browser.serverAddress=0.0.0.0
```

### For other web servers

Make sure your server allows being embedded in iframes by setting the appropriate headers:
```
Access-Control-Allow-Origin: *
X-Frame-Options: ALLOW
Content-Security-Policy: frame-ancestors 'self' *
```

### Use the "Open in New Tab" option

If you can't modify the server configuration, you can use the "Open in New Tab" button in the error message to open the chatbot in a new browser tab instead of an iframe.

### For development testing only

You can launch Chrome with security disabled for local testing (NOT recommended for general browsing):
```
chrome --disable-web-security --user-data-dir="/tmp/chrome-dev"
``` 
# Troubleshooting 403 Errors in RAG Chatbot Extension

If you're encountering "403 Forbidden" errors or seeing the "Error Loading Chatbot" message when trying to use the RAG Chatbot Chrome extension, follow these steps to resolve the issue.

## Quick Solutions

1. **Open in New Tab**: When the error message appears, click the "Open in New Tab" button to bypass iframe restrictions.

2. **Run Streamlit with the correct parameters**: If you're using Streamlit to host your RAG Chatbot, run it with these flags:
   ```bash
   streamlit run app.py --server.enableCORS=false --server.enableXsrfProtection=false --browser.serverAddress=0.0.0.0
   ```

## Understanding the Problem

The 403 Forbidden error typically occurs because of security restrictions that prevent websites from being embedded in iframes on different domains. This is a security feature called the Same-Origin Policy, which is enforced by web browsers.

The error could be caused by:

1. **CORS (Cross-Origin Resource Sharing) restrictions**: The server doesn't allow requests from different origins.
2. **X-Frame-Options header**: Set to `DENY` or `SAMEORIGIN`, preventing the page from being embedded in an iframe.
3. **Content Security Policy (CSP)**: Restricts which resources can be loaded and where they can be loaded from.

## Detailed Solutions

### Solution 1: Modify Streamlit Configuration

For Streamlit applications, create a `.streamlit/config.toml` file in your project with these settings:

```toml
[server]
enableCORS = false
enableXsrfProtection = false

[browser]
serverAddress = "0.0.0.0"
```

### Solution 2: Configure Your Web Server

If you're using a different web server, add these headers to your responses:

#### For Apache (.htaccess file):
```
Header set Access-Control-Allow-Origin "*"
Header set X-Frame-Options "ALLOW"
Header set Content-Security-Policy "frame-ancestors 'self' *"
```

#### For Nginx (nginx.conf):
```
add_header Access-Control-Allow-Origin "*";
add_header X-Frame-Options "ALLOW";
add_header Content-Security-Policy "frame-ancestors 'self' *";
```

#### For Express.js (Node.js):
```javascript
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('X-Frame-Options', 'ALLOW');
  res.header('Content-Security-Policy', "frame-ancestors 'self' *");
  next();
});
```

### Solution 3: Use a Proxy Server

If you can't modify the target server, consider setting up a proxy server that adds the appropriate headers:

1. Install a CORS proxy like `cors-anywhere`
2. Route your requests through this proxy
3. Update your URL in the extension to use the proxy URL

### Solution 4: For Development Testing Only

Run Chrome with security disabled (NOT recommended for general browsing):

**Windows:**
```
"C:\Program Files\Google\Chrome\Application\chrome.exe" --disable-web-security --user-data-dir="%TEMP%\chrome-dev"
```

**macOS:**
```
open -a "Google Chrome" --args --disable-web-security --user-data-dir="/tmp/chrome-dev"
```

**Linux:**
```
google-chrome --disable-web-security --user-data-dir="/tmp/chrome-dev"
```

## Still Having Issues?

If you've tried all these solutions and still encounter problems, you can:

1. Check the Chrome DevTools console (F12) for specific error messages
2. Try accessing the URL directly in a browser tab to ensure the service is running
3. Verify there are no network issues or firewall restrictions
4. Make sure your RAG Chatbot service is running and accessible from the same machine

For further assistance, please open an issue on the GitHub repository or contact the extension developer. 
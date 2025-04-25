// Check if bubble is already added
if (!document.getElementById('rag-chat-bubble')) {
    // Create chat bubble container
    const bubbleContainer = document.createElement('div');
    bubbleContainer.id = 'rag-chat-bubble';
    bubbleContainer.className = 'rag-chat-container';
    
    // Create chat bubble button
    const bubbleButton = document.createElement('button');
    bubbleButton.id = 'rag-chat-toggle';
    bubbleButton.className = 'rag-chat-bubble';
    bubbleButton.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
      </svg>
    `;
    
    // Create chat window
    const chatWindow = document.createElement('div');
    chatWindow.id = 'rag-chat-window';
    chatWindow.className = 'rag-chat-window';
    chatWindow.innerHTML = `
      <div class="rag-chat-header">
        <h2>RAG Chatbot</h2>
        <button id="rag-close-chat">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      <div id="rag-iframe-container">
        <iframe id="rag-iframe" src="" frameborder="0"></iframe>
        <div id="rag-error-message" style="display: none; padding: 20px; color: #e53e3e; text-align: center;">
          <h3>Error Loading Chatbot</h3>
          <p>The chatbot couldn't be loaded due to security restrictions. This may be due to:</p>
          <ul style="text-align: left; margin: 10px 0;">
            <li>CORS policy restrictions</li>
            <li>X-Frame-Options preventing iframe embedding</li>
            <li>Content Security Policy restrictions</li>
          </ul>
          <p>Possible solutions:</p>
          <ul style="text-align: left; margin: 10px 0;">
            <li>Make sure your server allows embedding (set appropriate CORS headers)</li>
            <li>If using Streamlit, run it with <code>--server.enableCORS=false --server.enableXsrfProtection=false --browser.serverAddress=0.0.0.0</code></li>
            <li>Try opening the chatbot in a new tab instead</li>
          </ul>
          <button id="rag-open-tab" style="background-color: #3b82f6; color: white; border: none; padding: 8px 16px; border-radius: 4px; margin-top: 10px; cursor: pointer;">
            Open in New Tab
          </button>
        </div>
      </div>
    `;
    
    // Append elements to container and then to body
    bubbleContainer.appendChild(bubbleButton);
    bubbleContainer.appendChild(chatWindow);
    document.body.appendChild(bubbleContainer);
    
    // Store the current URL
    let currentChatbotUrl = '';
    
    // Add event listeners
    document.getElementById('rag-chat-toggle').addEventListener('click', () => {
      // Retrieve last used URL from storage if available
      chrome.storage.local.get(['lastUsedUrl'], (result) => {
        const defaultUrl = result.lastUsedUrl || 'http://localhost:8501';
        const chatbotUrl = prompt('Enter RAG Chatbot URL (e.g., http://localhost:8501):', defaultUrl);
        
        if (chatbotUrl && chatbotUrl.trim() !== '') {
          currentChatbotUrl = chatbotUrl.trim();
          
          // Save URL to storage
          chrome.runtime.sendMessage({
            type: 'saveUrl',
            url: currentChatbotUrl
          });
          
          // Hide error message and show iframe
          document.getElementById('rag-error-message').style.display = 'none';
          document.getElementById('rag-iframe').style.display = 'block';
          
          // Set iframe src
          const iframe = document.getElementById('rag-iframe');
          iframe.src = currentChatbotUrl;
          document.getElementById('rag-chat-window').classList.add('open');
        }
      });
    });
    
    // Function to handle iframe loading errors
    function handleIframeError() {
      document.getElementById('rag-iframe').style.display = 'none';
      document.getElementById('rag-error-message').style.display = 'block';
      
      // Report error to background script
      if (currentChatbotUrl) {
        chrome.runtime.sendMessage({
          type: 'iframeError',
          url: currentChatbotUrl,
          timestamp: new Date().toISOString()
        }, (response) => {
          console.log('Error reported to background script');
        });
      }
    }
    
    // Add event listener for the "Open in New Tab" button
    document.body.addEventListener('click', (event) => {
      if (event.target.id === 'rag-open-tab') {
        if (currentChatbotUrl) {
          window.open(currentChatbotUrl, '_blank');
        }
      }
    });
    
    // Add load event listener to detect Content Security Policy violations
    document.getElementById('rag-iframe').addEventListener('load', function() {
      try {
        // Attempt to access iframe content - if it fails, it might be due to CORS/CSP
        const iframeContent = this.contentWindow.document;
        if (!iframeContent) {
          handleIframeError();
        }
      } catch (error) {
        // Security error occurred - probably CORS or CSP
        console.error('Error accessing iframe content:', error);
        handleIframeError();
      }
    });
    
    // Add error event listener for the iframe
    document.getElementById('rag-iframe').addEventListener('error', function(event) {
      console.error('Iframe loading error:', event);
      handleIframeError();
    });
    
    document.getElementById('rag-close-chat').addEventListener('click', () => {
      document.getElementById('rag-chat-window').classList.remove('open');
      document.getElementById('rag-iframe').src = '';
      currentChatbotUrl = '';
    });
}
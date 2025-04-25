// background.js - Service worker for the extension
chrome.runtime.onInstalled.addListener(() => {
    console.log('RAG Chatbot Extension installed');
    
    // Initialize storage with default settings
    chrome.storage.local.set({
      lastUsedUrl: 'http://127.0.0.1:8501/',
      errorCount: 0
    });
});

// Listen for messages from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'iframeError') {
      // Increment error count in storage
      chrome.storage.local.get(['errorCount'], (result) => {
        const newErrorCount = (result.errorCount || 0) + 1;
        chrome.storage.local.set({ errorCount: newErrorCount });
        console.log(`Iframe loading error #${newErrorCount}: ${message.url}`);
      });
      
      // Send response acknowledging the error
      sendResponse({ received: true });
    }
    
    if (message.type === 'saveUrl') {
      // Save the last used URL
      chrome.storage.local.set({ lastUsedUrl: message.url });
      sendResponse({ received: true });
    }
    
    // Return true to indicate we will send a response asynchronously
    return true;
});
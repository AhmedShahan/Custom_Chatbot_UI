document.getElementById('toggle-bubble').addEventListener('click', () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.scripting.executeScript({
        target: { tabId: tabs[0].id },
        function: toggleChatBubble,
      });
    });
  });
  
  function toggleChatBubble() {
    const bubble = document.getElementById('rag-chat-bubble');
    if (bubble) {
      // If bubble exists, toggle its visibility
      bubble.style.display = bubble.style.display === 'none' ? 'block' : 'none';
    } else {
      // If bubble doesn't exist yet, create it via content script
      const event = new CustomEvent('toggle-rag-bubble');
      document.dispatchEvent(event);
    }
  }
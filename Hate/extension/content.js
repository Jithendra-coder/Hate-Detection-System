// Content script for hate speech detection and highlighting
class HateSpeechDetector {
  constructor() {
    this.isEnabled = true;
    this.apiEndpoint = 'http://localhost:8000/predict'; // Your backend API
    this.highlightClass = 'hate-speech-highlight';
    this.processedElements = new WeakSet();

    this.init();
  }

  async init() {
    // Load user settings
    const settings = await chrome.storage.sync.get(['isEnabled', 'sensitivity', 'highlightColor']);
    this.isEnabled = settings.isEnabled !== false;
    this.sensitivity = settings.sensitivity || 0.7;
    this.highlightColor = settings.highlightColor || '#ff6b6b';

    if (this.isEnabled) {
      this.scanPage();
      this.observeChanges();
    }
  }

  // Main function to scan the entire page
  async scanPage() {
    const textNodes = this.getTextNodes(document.body);

    for (const node of textNodes) {
      if (!this.processedElements.has(node.parentElement)) {
        await this.processTextNode(node);
        this.processedElements.add(node.parentElement);
      }
    }
  }

  // Get all text nodes in the document
  getTextNodes(element) {
    const textNodes = [];
    const walker = document.createTreeWalker(
      element,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode: (node) => {
          // Skip script, style, and other non-content elements
          const parent = node.parentElement;
          if (!parent) return NodeFilter.FILTER_REJECT;

          const tagName = parent.tagName.toLowerCase();
          if (['script', 'style', 'noscript', 'meta'].includes(tagName)) {
            return NodeFilter.FILTER_REJECT;
          }

          // Skip if text is too short or only whitespace
          const text = node.textContent.trim();
          if (text.length < 3) return NodeFilter.FILTER_REJECT;

          return NodeFilter.FILTER_ACCEPT;
        }
      }
    );

    let currentNode;
    while (currentNode = walker.nextNode()) {
      textNodes.push(currentNode);
    }

    return textNodes;
  }

  // Process individual text nodes
  async processTextNode(node) {
    const text = node.textContent.trim();
    if (text.length < 10) return; // Skip very short text

    try {
      const prediction = await this.detectHateSpeech(text);

      if (prediction.isHateSpeech && prediction.confidence > this.sensitivity) {
        this.highlightText(node, prediction);
      }
    } catch (error) {
      console.warn('Hate speech detection failed:', error);
      // Fallback to simple keyword detection
      this.fallbackDetection(node, text);
    }
  }

  // Call your backend API for hate speech detection
  async detectHateSpeech(text) {
    try {
      const response = await fetch(this.apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text }),
        signal: AbortSignal.timeout(5000) // 5 second timeout
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const result = await response.json();
      return {
        isHateSpeech: result.prediction === 1 || result.label === 'hate',
        confidence: result.confidence || result.probability || 0.5,
        category: result.category || 'hate'
      };
    } catch (error) {
      throw new Error(`API call failed: ${error.message}`);
    }
  }

  // Fallback detection using simple keyword matching
  fallbackDetection(node, text) {
    const hateKeywords = [
      'hate', 'disgusting', 'awful', 'terrible', 'stupid', 'idiot', 
      'pathetic', 'useless', 'worthless', 'loser', 'scum'
    ];

    const lowerText = text.toLowerCase();
    const foundKeywords = hateKeywords.filter(keyword => lowerText.includes(keyword));

    if (foundKeywords.length > 0) {
      this.highlightText(node, {
        isHateSpeech: true,
        confidence: 0.6,
        category: 'fallback',
        keywords: foundKeywords
      });
    }
  }

  // Highlight detected hate speech
  highlightText(node, prediction) {
    const parent = node.parentElement;
    if (!parent || parent.classList.contains(this.highlightClass)) return;

    // Create highlight wrapper
    const highlight = document.createElement('span');
    highlight.className = this.highlightClass;
    highlight.style.backgroundColor = this.highlightColor;
    highlight.style.borderRadius = '3px';
    highlight.style.padding = '1px 2px';
    highlight.style.position = 'relative';
    highlight.style.cursor = 'help';

    // Add tooltip with detection info
    highlight.title = `Hate speech detected (${Math.round(prediction.confidence * 100)}% confidence)`;

    // Store detection data
    highlight.setAttribute('data-hate-speech', 'true');
    highlight.setAttribute('data-confidence', prediction.confidence);
    highlight.setAttribute('data-category', prediction.category);

    // Wrap the text node
    parent.replaceChild(highlight, node);
    highlight.appendChild(node);

    // Add click listener for more details
    highlight.addEventListener('click', (e) => {
      e.preventDefault();
      this.showHateSpeechDetails(highlight, prediction);
    });

    // Track highlighted elements
    this.trackHighlight(highlight, prediction);
  }

  // Show details about detected hate speech
  showHateSpeechDetails(element, prediction) {
    // Create or update info popup
    let popup = document.getElementById('hate-speech-info-popup');
    if (!popup) {
      popup = this.createInfoPopup();
    }

    const text = element.textContent;
    popup.innerHTML = `
      <div class="hate-speech-popup-content">
        <h3>Hate Speech Detected</h3>
        <p><strong>Text:</strong> "${text.substring(0, 100)}${text.length > 100 ? '...' : ''}"</p>
        <p><strong>Confidence:</strong> ${Math.round(prediction.confidence * 100)}%</p>
        <p><strong>Category:</strong> ${prediction.category}</p>
        <div class="popup-actions">
          <button id="report-false-positive">Report False Positive</button>
          <button id="close-popup">Close</button>
        </div>
      </div>
    `;

    // Position popup near the clicked element
    const rect = element.getBoundingClientRect();
    popup.style.top = (rect.bottom + window.scrollY + 5) + 'px';
    popup.style.left = Math.min(rect.left + window.scrollX, window.innerWidth - 300) + 'px';
    popup.style.display = 'block';

    // Add event listeners
    popup.querySelector('#close-popup').onclick = () => {
      popup.style.display = 'none';
    };

    popup.querySelector('#report-false-positive').onclick = () => {
      this.reportFalsePositive(text, prediction);
      popup.style.display = 'none';
    };
  }

  // Create info popup element
  createInfoPopup() {
    const popup = document.createElement('div');
    popup.id = 'hate-speech-info-popup';
    popup.style.cssText = `
      position: absolute;
      background: white;
      border: 2px solid #ff6b6b;
      border-radius: 8px;
      padding: 15px;
      max-width: 300px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      z-index: 10000;
      display: none;
      font-family: Arial, sans-serif;
      font-size: 14px;
      line-height: 1.4;
    `;
    document.body.appendChild(popup);
    return popup;
  }

  // Report false positive to improve the model
  async reportFalsePositive(text, prediction) {
    try {
      await fetch(this.apiEndpoint.replace('/predict', '/feedback'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text,
          predicted: prediction,
          actual: 'not_hate',
          feedback_type: 'false_positive'
        })
      });
      console.log('False positive reported successfully');
    } catch (error) {
      console.warn('Failed to report false positive:', error);
    }
  }

  // Track highlighted elements for statistics
  trackHighlight(element, prediction) {
    const data = {
      text: element.textContent.substring(0, 100),
      confidence: prediction.confidence,
      category: prediction.category,
      timestamp: Date.now(),
      url: window.location.href
    };

    chrome.storage.local.get(['detectionHistory'], (result) => {
      const history = result.detectionHistory || [];
      history.push(data);

      // Keep only last 100 detections
      if (history.length > 100) {
        history.shift();
      }

      chrome.storage.local.set({ detectionHistory: history });
    });
  }

  // Observe DOM changes for dynamic content
  observeChanges() {
    const observer = new MutationObserver((mutations) => {
      let hasTextChanges = false;

      mutations.forEach((mutation) => {
        if (mutation.type === 'childList') {
          mutation.addedNodes.forEach((node) => {
            if (node.nodeType === Node.TEXT_NODE || 
                (node.nodeType === Node.ELEMENT_NODE && node.textContent.trim())) {
              hasTextChanges = true;
            }
          });
        }
      });

      if (hasTextChanges) {
        // Debounce scanning to avoid excessive API calls
        clearTimeout(this.scanTimeout);
        this.scanTimeout = setTimeout(() => {
          this.scanPage();
        }, 1000);
      }
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
      characterData: false
    });
  }

  // Toggle extension on/off
  toggle(enabled) {
    this.isEnabled = enabled;

    if (enabled) {
      this.scanPage();
    } else {
      // Remove all highlights
      const highlights = document.querySelectorAll(`.${this.highlightClass}`);
      highlights.forEach(highlight => {
        const parent = highlight.parentNode;
        const textNode = highlight.firstChild;
        parent.replaceChild(textNode, highlight);
      });
    }
  }
}

// Initialize the hate speech detector when the page loads
let detector;

// Wait for DOM to be ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initDetector);
} else {
  initDetector();
}

function initDetector() {
  detector = new HateSpeechDetector();
}

// Listen for messages from popup/background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'toggle') {
    if (detector) {
      detector.toggle(request.enabled);
    }
    sendResponse({ success: true });
  } else if (request.action === 'scan') {
    if (detector) {
      detector.scanPage();
    }
    sendResponse({ success: true });
  } else if (request.action === 'getStats') {
    const highlights = document.querySelectorAll('.hate-speech-highlight');
    sendResponse({ 
      highlightCount: highlights.length,
      url: window.location.href 
    });
  }
});
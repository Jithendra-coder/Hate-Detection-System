// Popup script for hate speech detection extension
class HateSpeechPopup {
  constructor() {
    this.currentTab = null;
    this.init();
  }

  async init() {
    this.currentTab = await this.getCurrentTab();
    this.setupEventListeners();
    this.updateCurrentPageStatus();
  }

  async getCurrentTab() {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      return tab;
    } catch (error) {
      console.error('Failed to get current tab:', error);
      return null;
    }
  }

  setupEventListeners() {
    const scanButton = document.getElementById('scan-button');
    if (scanButton) {
      scanButton.addEventListener('click', () => {
        this.scanCurrentPage();
      });
    }
  }

  async updateCurrentPageStatus() {
    if (!this.currentTab) {
      document.getElementById('current-status').textContent = 'Unknown';
      document.getElementById('detection-count').textContent = '-';
      return;
    }
    try {
      // Try to get current page detection count
      const response = await chrome.tabs.sendMessage(this.currentTab.id, { action: 'getStats' });
      if (response && response.highlightCount !== undefined) {
        document.getElementById('detection-count').textContent = response.highlightCount;
        document.getElementById('current-status').textContent = 'Active';
        this.hideError();
      } else {
        document.getElementById('current-status').textContent = 'Not scanned';
        document.getElementById('detection-count').textContent = '-';
        this.hideError();
      }
    } catch (error) {
      document.getElementById('current-status').textContent = 'Unable to scan';
      document.getElementById('detection-count').textContent = '-';
      this.showError('Extension not active or unable to scan this page.');
    }
  }

  async scanCurrentPage() {
    if (!this.currentTab) return;
    this.setProgressBar(10); // Start progress

    try {
      this.hideError();
      this.setProgressBar(30);

      const response = await chrome.tabs.sendMessage(this.currentTab.id, {
        action: 'scan'
      });

      this.setProgressBar(70);

      if (response && response.success) {
        this.setProgressBar(100);
        setTimeout(() => {
          this.setProgressBar(0);
        }, 1200);

        await this.updateCurrentPageStatus();
      } else {
        throw new Error((response && response.error) || 'Scan failed');
      }
    } catch (error) {
      this.setProgressBar(0);
      this.showError('Unable to scan this page. ' + error.message);
    }
  }

  setProgressBar(percent) {
    const fill = document.getElementById('scan-progress-fill');
    if (fill) {
      fill.style.width = Math.max(0, Math.min(percent, 100)) + '%';
    }
  }

  showError(message) {
    const errorDiv = document.getElementById('scan-error-message');
    if (errorDiv) {
      errorDiv.style.display = 'block';
      errorDiv.textContent = message || 'An error occurred during scanning.';
    }
  }

  hideError() {
    const errorDiv = document.getElementById('scan-error-message');
    if (errorDiv) {
      errorDiv.style.display = 'none';
      errorDiv.textContent = '';
    }
  }
}

// Initialize popup when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  new HateSpeechPopup();
});

// Background script for hate speech detection extension
class HateSpeechBackground {
  constructor() {
    this.init();
  }

  init() {
    // Handle extension installation
    chrome.runtime.onInstalled.addListener((details) => {
      this.onInstalled(details);
    });

    // Handle messages from content scripts and popup
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      this.handleMessage(message, sender, sendResponse);
      return true; // Keep message channel open for async responses
    });

    // Handle browser action clicks
    chrome.action.onClicked.addListener((tab) => {
      this.toggleExtension(tab);
    });

    // Monitor tab updates to scan new pages
    chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
      if (changeInfo.status === 'complete' && tab.url && !tab.url.startsWith('chrome://')) {
        this.scanNewPage(tab);
      }
    });
  }

  // Handle extension installation and updates
  async onInstalled(details) {
    console.log('Hate Speech Detector installed:', details.reason);

    // Set default settings
    const defaultSettings = {
      isEnabled: true,
      sensitivity: 0.7,
      highlightColor: '#ff6b6b',
      autoScan: true,
      showNotifications: true
    };

    // Initialize storage with defaults
    await chrome.storage.sync.set(defaultSettings);

    // Show welcome page on first install
    if (details.reason === 'install') {
      chrome.tabs.create({
        url: chrome.runtime.getURL('welcome.html')
      });
    }
  }

  // Handle messages from different parts of the extension
  async handleMessage(message, sender, sendResponse) {
    try {
      switch (message.action) {
        case 'getSettings':
          const settings = await chrome.storage.sync.get();
          sendResponse({ success: true, settings });
          break;

        case 'updateSettings':
          await chrome.storage.sync.set(message.settings);
          await this.broadcastSettingsUpdate(message.settings);
          sendResponse({ success: true });
          break;

        case 'getStats':
          const stats = await this.getDetectionStats();
          sendResponse({ success: true, stats });
          break;

        case 'exportData':
          const data = await this.exportUserData();
          sendResponse({ success: true, data });
          break;

        case 'reportIssue':
          await this.reportIssue(message.data);
          sendResponse({ success: true });
          break;

        case 'scanPage':
          if (sender.tab) {
            await this.scanTabContent(sender.tab.id);
          }
          sendResponse({ success: true });
          break;

        case 'scan-progress':
          // Relay scan progress message to the tab to be picked by popup or content scripts
          if (sender.tab && message.progress !== undefined) {
            chrome.tabs.sendMessage(sender.tab.id, message);
          }
          sendResponse({ success: true });
          break;

        default:
          sendResponse({ success: false, error: 'Unknown action' });
      }
    } catch (error) {
      console.error('Background script error:', error);
      sendResponse({ success: false, error: error.message });
    }
  }

  // Toggle extension on/off for current tab
  async toggleExtension(tab) {
    try {
      const settings = await chrome.storage.sync.get(['isEnabled']);
      const newState = !settings.isEnabled;

      await chrome.storage.sync.set({ isEnabled: newState });

      // Update icon based on state
      chrome.action.setIcon({
        path: newState
          ? {
              "16": "icons/icon-16.png",
              "32": "icons/icon-32.png",
              "48": "icons/icon-48.png",
              "128": "icons/icon-128.png"
            }
          : {
              "16": "icons/icon-disabled-16.png",
              "32": "icons/icon-disabled-32.png",
              "48": "icons/icon-disabled-48.png",
              "128": "icons/icon-disabled-128.png"
            }
      });

      // Update badge
      chrome.action.setBadgeText({
        text: newState ? '' : 'OFF',
        tabId: tab.id
      });

      // Notify content script
      chrome.tabs.sendMessage(tab.id, {
        action: 'toggle',
        enabled: newState
      });
    } catch (error) {
      console.error('Failed to toggle extension:', error);
    }
  }

  // Scan new pages automatically
  async scanNewPage(tab) {
    try {
      const settings = await chrome.storage.sync.get(['isEnabled', 'autoScan']);
      if (settings.isEnabled && settings.autoScan) {
        // Wait for page to fully load
        setTimeout(async () => {
          try {
            await chrome.tabs.sendMessage(tab.id, { action: 'scan' });
          } catch (error) {
            // Content script might not be loaded yet
            console.log('Content script not ready for tab:', tab.id);
          }
        }, 2000);
      }
    } catch (error) {
      console.error('Failed to scan new page:', error);
    }
  }

  // Broadcast settings updates to all tabs
  async broadcastSettingsUpdate(newSettings) {
    try {
      const tabs = await chrome.tabs.query({});
      for (const tab of tabs) {
        try {
          chrome.tabs.sendMessage(tab.id, {
            action: 'settingsUpdated',
            settings: newSettings
          });
        } catch {
          // Tab might not have content script loaded
          continue;
        }
      }
    } catch (error) {
      console.error('Failed to broadcast settings:', error);
    }
  }

  // Get detection statistics
  async getDetectionStats() {
    try {
      const result = await chrome.storage.local.get(['detectionHistory', 'totalDetections']);
      const history = result.detectionHistory || [];
      const total = result.totalDetections || 0;

      const today = new Date().toDateString();
      const todayDetections = history.filter(item =>
        new Date(item.timestamp).toDateString() === today);

      const stats = {
        total: total,
        today: todayDetections.length,
        thisWeek: this.getWeeklyDetections(history),
        categories: this.getCategoryStats(history),
        topSites: this.getTopSites(history)
      };

      return stats;
    } catch (error) {
      console.error('Failed to get stats:', error);
      return { total: 0, today: 0, thisWeek: 0, categories: {}, topSites: [] };
    }
  }

  // Calculate weekly detections
  getWeeklyDetections(history) {
    const weekAgo = Date.now() - (7 * 24 * 60 * 60 * 1000);
    return history.filter(item => item.timestamp > weekAgo).length;
  }

  // Get statistics by category
  getCategoryStats(history) {
    const categories = {};
    history.forEach(item => {
      const category = item.category || 'unknown';
      categories[category] = (categories[category] || 0) + 1;
    });
    return categories;
  }

  // Get top sites with most detections
  getTopSites(history) {
    const sites = {};
    history.forEach(item => {
      try {
        const hostname = new URL(item.url).hostname;
        sites[hostname] = (sites[hostname] || 0) + 1;
      } catch {
        // Invalid URL, skip it
      }
    });

    return Object.entries(sites)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .map(([site, count]) => ({ site, count }));
  }

  // Export user data for backup/analysis
  async exportUserData() {
    try {
      const syncData = await chrome.storage.sync.get();
      const localData = await chrome.storage.local.get();

      return {
        settings: syncData,
        history: localData.detectionHistory || [],
        exportDate: new Date().toISOString(),
        version: chrome.runtime.getManifest().version
      };
    } catch (error) {
      console.error('Failed to export data:', error);
      throw error;
    }
  }

  // Report issues or feedback
  async reportIssue(data) {
    try {
      // In a real implementation, you would send this to your server
      console.log('Issue reported:', data);

      // Store locally for now
      const issues = await chrome.storage.local.get(['reportedIssues']);
      const reportedIssues = issues.reportedIssues || [];

      reportedIssues.push({
        ...data,
        timestamp: Date.now(),
        version: chrome.runtime.getManifest().version
      });

      await chrome.storage.local.set({ reportedIssues });

      // Show notification
      chrome.notifications.create({
        type: 'basic',
        iconUrl: 'icons/icon-48.png',
        title: 'Issue Reported',
        message: 'Thank you for your feedback. We will look into it.'
      });
    } catch (error) {
      console.error('Failed to report issue:', error);
      throw error;
    }
  }

  // Scan specific tab content
  async scanTabContent(tabId) {
    try {
      chrome.tabs.sendMessage(tabId, { action: 'scan' });
    } catch (error) {
      console.error('Failed to scan tab content:', error);
    }
  }
}

// Initialize background script
new HateSpeechBackground();

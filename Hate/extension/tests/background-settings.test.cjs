const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

test("installation fills missing defaults without replacing saved choices", async () => {
  let onInstalled;
  const settings = { isEnabled: false, sensitivity: 0.93, autoScan: true };
  const chrome = {
    runtime: {
      onInstalled: { addListener: (listener) => { onInstalled = listener; } },
      onMessage: { addListener() {} },
    },
    action: { onClicked: { addListener() {} } },
    tabs: { onUpdated: { addListener() {} } },
    storage: {
      sync: {
        get: async () => ({ ...settings }),
        set: async (updates) => Object.assign(settings, updates),
      },
    },
  };
  const source = fs.readFileSync(path.join(__dirname, "..", "background.js"), "utf8");
  vm.runInNewContext(source, { chrome, console, URL, Date, setTimeout });

  onInstalled({ reason: "update" });
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.equal(settings.isEnabled, false);
  assert.equal(settings.sensitivity, 0.93);
  assert.equal(settings.autoScan, true);
  assert.equal(settings.highlightColor, "#ff6b6b");
  assert.equal(settings.showNotifications, true);
});

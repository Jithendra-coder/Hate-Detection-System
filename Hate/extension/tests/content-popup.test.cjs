const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

test("popup renders page text and model category as text, never markup", () => {
  const source = fs.readFileSync(path.join(__dirname, "..", "content.js"), "utf8");
  const fields = new Map();
  const popup = {
    style: {},
    innerHTML: "",
    querySelector(selector) {
      if (!fields.has(selector)) fields.set(selector, { textContent: "", onclick: null });
      return fields.get(selector);
    },
  };
  const document = {
    readyState: "loading",
    addEventListener() {},
    getElementById: () => popup,
  };
  const context = {
    document,
    window: { scrollY: 0, scrollX: 0, innerWidth: 1000 },
    chrome: { runtime: { onMessage: { addListener() {} } } },
    console,
  };
  vm.runInNewContext(`${source}\nglobalThis.popupMethod = HateSpeechDetector.prototype.showHateSpeechDetails;`, context);

  const hostileText = '<img src=x onerror=alert(1)>';
  const hostileCategory = '<script>alert(1)</script>';
  context.popupMethod.call(
    { reportFalsePositive() {} },
    { textContent: hostileText, getBoundingClientRect: () => ({ bottom: 0, left: 0 }) },
    { confidence: 0.95, category: hostileCategory },
  );

  assert.equal(fields.get("#hate-popup-text").textContent, `"${hostileText}"`);
  assert.equal(fields.get("#hate-popup-category").textContent, hostileCategory);
  assert.equal(popup.innerHTML.includes(hostileText), false);
  assert.equal(popup.innerHTML.includes(hostileCategory), false);
});

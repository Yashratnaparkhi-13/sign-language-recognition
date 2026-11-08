const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

const utterances = [];
let cancelCount = 0;
const listeners = {};
const status = { textContent: "", className: "" };

class MockUtterance {
  constructor(text) {
    this.text = text;
    utterances.push(this);
  }
}

const window = {
  parent: {
    postMessage(message) {
      if (message.type === "streamlit:setComponentValue") {
        status.componentValue = message.value;
      }
    },
  },
  speechSynthesis: {
    speaking: false,
    cancel() {
      cancelCount += 1;
      this.speaking = false;
    },
    speak(utterance) {
      this.speaking = true;
      utterance.onstart();
      this.speaking = false;
      utterance.onend();
    },
  },
  SpeechSynthesisUtterance: MockUtterance,
  addEventListener(name, callback) {
    listeners[name] = callback;
  },
};

const context = {
  window,
  document: { getElementById: () => status },
  SpeechSynthesisUtterance: MockUtterance,
};
const script = fs.readFileSync(
  path.join(__dirname, "../web_app/speech/frontend/speech.js"),
  "utf8",
);
vm.runInNewContext(script, context);

function render(args) {
  listeners.message({ data: { type: "streamlit:render", args } });
}

const base = {
  enabled: true,
  word_event_id: 0,
  completed_word: "",
  replay_word_id: 0,
  sentence_event_id: 0,
  sentence: "",
};

render(base);
assert.equal(utterances.length, 0, "initial render must not speak");
render({ ...base, word_event_id: 1, completed_word: "YASH" });
assert.equal(utterances.length, 1);
assert.equal(utterances[0].text, "Yash.");
render({ ...base, word_event_id: 1, completed_word: "YASH" });
assert.equal(utterances.length, 1, "same event must not be repeated");
render({ ...base, word_event_id: 1, completed_word: "YASH", replay_word_id: 1 });
assert.equal(utterances.length, 2, "replay action must speak once");
render({ ...base, word_event_id: 1, completed_word: "YASH", replay_word_id: 1, sentence_event_id: 1, sentence: "YASH HOME" });
assert.equal(utterances.length, 3);
assert.equal(utterances[2].text, "Yash Home.");
render({ ...base, enabled: false, word_event_id: 1, replay_word_id: 1, sentence_event_id: 1 });
assert.equal(status.componentValue, "muted");
assert.ok(cancelCount >= 1);
render({ ...base, enabled: true, word_event_id: 1, replay_word_id: 1, sentence_event_id: 1, sentence: "YASH HOME" });
assert.equal(utterances.length, 3, "unmute must not replay old events");

console.log("Browser speech tests passed.");

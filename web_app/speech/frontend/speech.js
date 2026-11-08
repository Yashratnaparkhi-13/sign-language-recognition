(function () {
  const statusElement = document.getElementById("status");
  let lastWordEvent = null;
  let lastReplayWordEvent = null;
  let lastSentenceEvent = null;
  let lastStatus = "";
  let initialized = false;

  function send(type, value) {
    window.parent.postMessage(
      {
        isStreamlitMessage: true,
        type: type,
        ...(value === undefined ? {} : { value: value, dataType: "json" })
      },
      "*"
    );
  }

  function setStatus(status) {
    if (status === lastStatus) return;
    lastStatus = status;
    statusElement.textContent =
      status === "speaking" ? "Speaking on this device…" :
      status === "muted" ? "Browser voice is off" :
      status === "error" ? "Browser speech is unavailable" :
      "Voice ready";
    statusElement.className = status;
    send("streamlit:setComponentValue", status);
  }

  function spokenForm(text) {
    const normalized = String(text || "").trim();
    if (!normalized) return "";
    const sentenceCase = normalized.toLocaleLowerCase().replace(
      /(^|[\s.!?])([a-z])/g,
      (_, prefix, letter) => prefix + letter.toLocaleUpperCase()
    );
    return /[.!?]$/.test(sentenceCase) ? sentenceCase : sentenceCase + ".";
  }

  function speak(text) {
    const phrase = spokenForm(text);
    if (!phrase) return;
    if (!("speechSynthesis" in window) || !("SpeechSynthesisUtterance" in window)) {
      setStatus("error");
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(phrase);
    utterance.onstart = () => setStatus("speaking");
    utterance.onend = () => setStatus("idle");
    utterance.onerror = (event) => {
      if (event.error !== "canceled" && event.error !== "interrupted") {
        setStatus("error");
      }
    };
    window.speechSynthesis.speak(utterance);
  }

  function handleRender(event) {
    const args = (event.data && event.data.args) || {};
    if (!args.enabled) {
      if ("speechSynthesis" in window) window.speechSynthesis.cancel();
      lastWordEvent = Number(args.word_event_id || 0);
      lastReplayWordEvent = Number(args.replay_word_id || 0);
      lastSentenceEvent = Number(args.sentence_event_id || 0);
      initialized = true;
      setStatus("muted");
      return;
    }

    if (!("speechSynthesis" in window)) {
      setStatus("error");
      return;
    }
    if (!window.speechSynthesis.speaking) setStatus("idle");
    const wordEvent = Number(args.word_event_id || 0);
    const replayWordEvent = Number(args.replay_word_id || 0);
    const sentenceEvent = Number(args.sentence_event_id || 0);

    if (!initialized) {
      lastWordEvent = wordEvent;
      lastReplayWordEvent = replayWordEvent;
      lastSentenceEvent = sentenceEvent;
      initialized = true;
      return;
    }

    if (wordEvent !== lastWordEvent) {
      lastWordEvent = wordEvent;
      if (args.completed_word) speak(args.completed_word);
    }
    if (replayWordEvent !== lastReplayWordEvent) {
      lastReplayWordEvent = replayWordEvent;
      if (args.completed_word) speak(args.completed_word);
    }
    if (sentenceEvent !== lastSentenceEvent) {
      lastSentenceEvent = sentenceEvent;
      if (args.sentence) speak(args.sentence);
    }
  }

  window.addEventListener("message", function (event) {
    if (event.data && event.data.type === "streamlit:render") {
      handleRender(event);
    }
  });

  window.parent.postMessage(
    { isStreamlitMessage: true, type: "streamlit:componentReady", apiVersion: 1 },
    "*"
  );
  window.parent.postMessage(
    { isStreamlitMessage: true, type: "streamlit:setFrameHeight", height: 34 },
    "*"
  );
})();

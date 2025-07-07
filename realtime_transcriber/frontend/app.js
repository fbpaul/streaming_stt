const ws = new WebSocket("ws://localhost:8000/ws");
const resultsContainer = document.getElementById("results");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  const block = document.createElement("div");
  block.className = "transcript-block";

  if (data.type === "interim") {
    const interimLine = document.createElement("div");
    interimLine.className = "interim";
    interimLine.textContent = `暫定：${data.text}`;
    block.appendChild(interimLine);
  }

  if (data.type === "final") {
    const speaker = document.createElement("div");
    speaker.className = "speaker";
    speaker.textContent = `[${data.speaker}] ${formatTime(data.start)} - ${formatTime(data.end)}`;

    const text = document.createElement("div");
    text.className = "text";
    text.textContent = data.text;

    block.appendChild(speaker);
    block.appendChild(text);
  }

  resultsContainer.appendChild(block);
  resultsContainer.scrollTop = resultsContainer.scrollHeight;
};

function formatTime(seconds) {
  const mins = Math.floor(seconds / 60).toString().padStart(2, '0');
  const secs = Math.floor(seconds % 60).toString().padStart(2, '0');
  return `${mins}:${secs}`;
}

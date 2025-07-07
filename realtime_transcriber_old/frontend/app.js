const ws = new WebSocket("ws://localhost:8000/ws");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "interim") {
    document.getElementById("interim").textContent = data.text;
  } else if (data.type === "final") {
    document.getElementById("final").textContent += data.text + "\n";
  } else if (data.type === "speaker") {
    document.getElementById("speaker").textContent = JSON.stringify(data.segments, null, 2);
  }
};

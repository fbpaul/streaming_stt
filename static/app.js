const socket = io(); // 自動連接到伺服器上的 socket.io

const interimContainer = document.getElementById("interim");
const finalContainer = document.getElementById("final");

let currentLine = "";  // 累積暫定句用

// 播放音檔
const audio = new Audio("/static/test.mp3");
audio.play();

// 接收訊息
socket.on("transcription", (data) => {
  if (data.type === "interim") {
    currentLine += data.text + " ";
    interimContainer.innerText = "⏳ [暫定] " + currentLine.trim();
  } else if (data.type === "final") {
    const text = `✅ [修正] (${data.speaker}) ${data.start}~${data.end}s: ${data.text}`;
    
    // 新增 final 結果
    const finalElem = document.createElement("div");
    finalElem.innerText = text;
    finalContainer.appendChild(finalElem);

    // 清空暫定顯示
    interimContainer.innerText = "";
    currentLine = "";
  }
});

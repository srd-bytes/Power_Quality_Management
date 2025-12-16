const chatBox = document.getElementById("chat-box");
const input = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");

let history = ""; // ✅ Store history across messages

function appendMessage(sender, message) {
    const div = document.createElement("div");
    div.className = sender;
    div.innerHTML = `<strong>${sender === 'user' ? 'You' : 'Agent'}:</strong> ${message}`;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function sendMessage() {
    const message = input.value.trim();
    if (message === "") return;

    appendMessage('user', message);
    input.value = "";

    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, history })  // ✅ Send history too
    })
    .then(response => response.json())
    .then(data => {
        console.log("📥 Received response from backend:", data);
        appendMessage('agent', data.response);
        history = data.history; // ✅ Update local history
        console.log("🧠 Updated local history:", history);
    });
    

}

sendButton.addEventListener("click", sendMessage);

input.addEventListener("keypress", function (e) {
    if (e.key === "Enter") sendMessage();
});

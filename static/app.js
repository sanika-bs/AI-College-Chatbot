const SUGGESTIONS = {
  Admissions: ["What is the admission procedure?", "How can I join this college?", "What is the eligibility criteria?"],
  Courses: ["What courses do you offer?", "Which departments are available?"],
  Fees: ["What is the fee structure?", "Are there any scholarships?"],
  Internships: ["Are internships provided?", "Is internship mandatory?"],
  Placements: ["What is the average package?", "Which companies visit for placements?"],
  "Campus & Exams": ["Is hostel facility available?", "What are the library timings?", "When are the exams?", "How can I contact the college?"]
};
const $ = id => document.getElementById(id);
const log = $("log"), input = $("input"), sendBtn = $("send");
let history = JSON.parse(sessionStorage.getItem("chat") || "[]");   // conversation history for this session

function render(m) {
  const d = document.createElement("div");
  if (m.role === "user") { d.className = "msg user"; d.textContent = m.text; }
  else {
    d.className = "msg bot" + (m.fallback ? " fb" : "");
    if (m.confidence !== undefined) {
      const meta = document.createElement("div"); meta.className = "meta";
      const pct = (m.confidence * 100).toFixed(1);
      meta.innerHTML = `<span class="tag ${m.fallback ? "warn" : ""}">${m.fallback ? "Fallback &middot; low confidence" : "Intent: " + m.intent.replace("_", " ")}</span>` +
        `<span class="tag">Confidence ${pct}% <span class="bar"><i style="width:${pct}%"></i></span></span>` +
        (m.fallback ? `<span class="tag warn">closest guess: ${m.guess.replace("_", " ")}</span>` : "");
      d.appendChild(meta);
    }
    const t = document.createElement("div"); t.textContent = m.text; d.appendChild(t);
  }
  log.appendChild(d); log.scrollTop = log.scrollHeight; return d;
}
function save() { sessionStorage.setItem("chat", JSON.stringify(history)); }
function add(m) { history.push(m); save(); render(m); }
function welcome() { add({ role: "bot", text: "Hi! I'm the SIT student assistant. Ask me about admissions, courses, fees, internships, placements, hostel and more — or tap a suggested question." }); }

async function ask(q) {
  q = (q || "").trim(); if (!q) return;
  input.value = ""; add({ role: "user", text: q }); sendBtn.disabled = true;
  $("side").classList.remove("open");
  const loader = render({ role: "bot", text: "" }); loader.innerHTML = '<span class="dots"><span></span><span></span><span></span></span>';
  try {
    const res = await fetch("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q }) });
    const j = await res.json(); loader.remove();
    if (!res.ok) add({ role: "bot", text: j.error || "Something went wrong.", fallback: true });
    else add({ role: "bot", text: j.answer, intent: j.intent, confidence: j.confidence, fallback: j.is_fallback, guess: j.best_guess });
  } catch (e) { loader.remove(); add({ role: "bot", text: "Can't reach the server right now. Please check your connection and try again.", fallback: true }); }
  sendBtn.disabled = false; input.focus();
}
sendBtn.onclick = () => ask(input.value);
input.addEventListener("keydown", e => { if (e.key === "Enter") ask(input.value); });
$("clearBtn").onclick = () => { history = []; save(); log.innerHTML = ""; welcome(); };
$("menuBtn").onclick = () => $("side").classList.toggle("open");
$("suggestions").innerHTML = Object.entries(SUGGESTIONS).map(([c, qs]) => `<div class="cat">${c}</div>` + qs.map(q => `<button class="chip">${q}</button>`).join("")).join("");
document.querySelectorAll(".chip").forEach(b => b.onclick = () => ask(b.textContent));
fetch("/health").then(r => r.json()).then(j => $("status").textContent = "online").catch(() => $("status").textContent = "offline");
history.length ? history.forEach(render) : welcome();

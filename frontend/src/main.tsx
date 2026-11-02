import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { BookOpen, ChevronDown, FileText, LoaderCircle, Plus, Send, SlidersHorizontal, Sparkles } from "lucide-react";
import "./styles.css";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1";

type Citation = { id: string; title: string; section: string; source: string; excerpt: string; score: number };
type Response = { answer: string; mode: "document_rag" | "structured_api" | "clarification"; citations: Citation[]; structured_data?: Record<string, any>; metadata: Record<string, any> };
type Message = { role: "user" | "assistant"; content: string; response?: Response };

const examples = [
  "What are federal obligations and how are they different from outlays?",
  "How much did the Department of Education obligate in fiscal year 2024?",
  "What does the USAspending API provide?",
];

function App() {
  const [conversationId, setConversationId] = useState<string>();
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [agency, setAgency] = useState("");
  const [fiscalYear, setFiscalYear] = useState("2024");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => { void startConversation(); }, []);

  async function startConversation() {
    const res = await fetch(`${API}/conversations`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) });
    const data = await res.json();
    setConversationId(data.id);
    setMessages([]);
  }

  async function ask(value = question) {
    if (!value.trim() || !conversationId || loading) return;
    setQuestion(""); setError(""); setLoading(true);
    setMessages((current) => [...current, { role: "user", content: value }]);
    try {
      const res = await fetch(`${API}/conversations/${conversationId}/messages`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: value, agency, fiscal_year: Number(fiscalYear) || null }) });
      if (!res.ok) throw new Error((await res.json()).detail ?? "The research request failed.");
      const response: Response = await res.json();
      setMessages((current) => [...current, { role: "assistant", content: response.answer, response }]);
    } catch (err) { setError(err instanceof Error ? err.message : "The research request failed."); }
    finally { setLoading(false); }
  }

  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><Sparkles size={18} /></div><span>FederalScope</span></div>
      <button className="new-button" onClick={() => void startConversation()}><Plus size={17} /> New research</button>
      <div className="side-label">ABOUT THIS DEMO</div>
      <p className="side-copy">A research workspace for exploring federal spending using official documents and structured USAspending data.</p>
      <div className="side-label">SYSTEM</div>
      <div className="system-row"><span className="status-dot" /> Retrieval service online</div>
      <div className="system-row"><BookOpen size={15} /> Evidence-first answers</div>
      <div className="side-label">NUMERIC LOOKUP</div>
      <p className="side-hint">Add an agency for exact spending totals. Narrative questions can leave this blank.</p>
      <label className="field-label">Agency <input value={agency} onChange={(event) => setAgency(event.target.value)} placeholder="Optional: e.g. NASA" /></label>
      <label className="field-label">Fiscal year <input value={fiscalYear} onChange={(event) => setFiscalYear(event.target.value)} inputMode="numeric" /></label>
      <div className="side-footer">Built as a full-stack RAG product<br />with a transparent answer path.</div>
    </aside>
    <main className="workspace">
      <header className="topbar"><div><div className="eyebrow">RESEARCH WORKSPACE</div><h1>Federal spending assistant</h1></div><button className="icon-button"><SlidersHorizontal size={18} /></button></header>
      <section className="content">
        {messages.length === 0 ? <div className="empty-state"><div className="hero-icon"><Sparkles size={24} /></div><h2>Ask a question. Inspect the evidence.</h2><p>Get grounded explanations from the indexed corpus, or exact figures from USAspending.gov.</p><div className="example-grid">{examples.map((item) => <button key={item} className="example" onClick={() => void ask(item)}><span>{item}</span><Send size={15} /></button>)}</div></div> : <div className="thread">{messages.map((message, index) => <MessageView key={index} message={message} />)}{loading && <div className="loading"><LoaderCircle className="spin" size={18} /> Searching sources and preparing a grounded answer...</div>}</div>}
        {error && <div className="error">{error}</div>}
      </section>
      <div className="composer-wrap"><form className="composer" onSubmit={(event) => { event.preventDefault(); void ask(); }}><textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about federal spending..." rows={1} /><button type="submit" disabled={!question.trim() || loading}><Send size={18} /></button></form><div className="composer-note">Answers cite indexed evidence. Numeric values come directly from USAspending.gov when available.</div></div>
    </main>
  </div>;
}

function MessageView({ message }: { message: Message }) {
  if (message.role === "user") return <div className="message user-message"><div className="avatar user-avatar">You</div><div className="bubble user-bubble">{message.content}</div></div>;
  const response = message.response;
  return <div className="message"><div className="avatar assistant-avatar"><Sparkles size={15} /></div><div className="answer-area"><div className="answer-text">{plainText(message.content)}</div>{response?.structured_data && <StructuredCard data={response.structured_data} />}{response?.citations.length ? <div className="evidence"><div className="section-heading"><span>Evidence used</span><span className="mode-pill">{response.mode === "structured_api" ? "STRUCTURED API" : "DOCUMENT RAG"}</span></div>{response.citations.map((citation, index) => <details className="citation" key={citation.id}><summary><span className="citation-number">{index + 1}</span><span><strong>{citation.title}</strong><small>{citation.section} · relevance {citation.score.toFixed(2)}</small></span><ChevronDown size={15} /></summary><p>{citation.excerpt}</p></details>)}</div> : null}{response?.metadata && <div className="metadata">{response.metadata.retrieved_chunks ? `${response.metadata.retrieved_chunks} sources · ` : ""}{response.metadata.latency_ms} ms · {response.metadata.model}</div>}</div></div>;
}

function plainText(value: string) {
  return value.replace(/\[([^\]]+)\]\([^)]*\)/g, "$1").replace(/`([^`]+)`/g, "$1").replace(/\*\*([^*]+)\*\*/g, "$1");
}

function StructuredCard({ data }: { data: Record<string, any> }) {
  const values = [["Budgetary resources", data.budgetary_resources], ["Obligations", data.obligations], ["Outlays", data.outlays]];
  return <div className="data-card"><div className="data-card-head"><div><div className="eyebrow">AUTHORITATIVE VALUES</div><strong>{data.agency}</strong></div><span>FY {data.fiscal_year}</span></div><div className="data-grid">{values.map(([label, value]) => <div key={label as string}><small>{label}</small><strong>{typeof value === "number" ? `$${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : "Unavailable"}</strong></div>)}</div></div>;
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);

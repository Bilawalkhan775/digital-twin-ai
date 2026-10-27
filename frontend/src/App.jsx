
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

// ==========================================
// CONFIGURATION
// ==========================================

const API_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000";

const STORAGE_KEY = "digital-twin-access-token";

// ==========================================
// STARTER QUESTIONS
// ==========================================

const STARTER_QUESTIONS = [
  "What projects has Bilawal worked on?",
  "What are his technical skills?",
  "Tell me about his work experience.",
  "What backend development experience does he have?",
];

// ==========================================
// MAIN APPLICATION
// ==========================================

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const [accessToken, setAccessToken] = useState(
    () => localStorage.getItem(STORAGE_KEY) || null
  );

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [restoringHistory, setRestoringHistory] = useState(
    () => Boolean(localStorage.getItem(STORAGE_KEY))
  );

  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const activeRequest = useRef(null);
  const accessTokenRef = useRef(accessToken);
  const chatGenerationRef = useRef(0);
  const loadingRef = useRef(false);

  // ==========================================
  // TOKEN MANAGEMENT
  // ==========================================

  function updateAccessToken(token) {
    accessTokenRef.current = token;
    setAccessToken(token);

    if (token) {
      localStorage.setItem(STORAGE_KEY, token);
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  }

  // ==========================================
  // AUTO SCROLL
  // ==========================================

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading, restoringHistory]);

  // ==========================================
  // RESTORE CONVERSATION HISTORY
  // ==========================================

  useEffect(() => {
    const savedToken = localStorage.getItem(STORAGE_KEY);

    if (!savedToken) {
      return;
    }

    const controller = new AbortController();
    const generation = chatGenerationRef.current;

    async function restoreHistory() {
      try {
        const response = await fetch(
          `${API_URL}/api/history`,
          {
            method: "GET",
            headers: {
              Authorization: `Bearer ${savedToken}`,
            },
            signal: controller.signal,
          }
        );

        if (controller.signal.aborted) return;

        if (generation !== chatGenerationRef.current) {
          return;
        }

        // Conversation expired or invalid.
        if (
          response.status === 401 ||
          response.status === 410
        ) {
          updateAccessToken(null);
          setMessages([]);
          return;
        }

        if (!response.ok) {
          throw new Error(
            `Could not restore chat history (HTTP ${response.status}).`
          );
        }

        const data = await response.json();

        if (controller.signal.aborted) return;

        if (generation !== chatGenerationRef.current) {
          return;
        }

        // Restore only valid messages.
        const restoredMessages = (
          Array.isArray(data.messages)
            ? data.messages
            : []
        ).filter(
          (message) =>
            (message.role === "user" ||
              message.role === "assistant") &&
            typeof message.content === "string"
        );

        setMessages(restoredMessages);

        if (data.access_token) {
          updateAccessToken(data.access_token);
        }
      } catch (err) {
        if (err.name !== "AbortError") {
          console.error(
            "History restoration failed:",
            err
          );

          setError(
            "Unable to restore your previous conversation. " +
              "Please check your connection and refresh."
          );
        }
      } finally {
        if (
          !controller.signal.aborted &&
          generation === chatGenerationRef.current
        ) {
          setRestoringHistory(false);
        }
      }
    }

    restoreHistory();

    return () => controller.abort();
  }, []);

  // ==========================================
  // CLEANUP ON UNMOUNT
  // ==========================================

  useEffect(() => {
    return () => {
      activeRequest.current?.abort();
    };
  }, []);

  // ==========================================
  // SEND MESSAGE
  // ==========================================

  async function sendMessage(questionText) {
    const question = questionText.trim();

    if (
      !question ||
      loadingRef.current ||
      restoringHistory
    ) {
      return;
    }

    loadingRef.current = true;

    const generation = chatGenerationRef.current;
    const token = accessTokenRef.current;

    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        content: question,
      },
    ]);

    setInput("");
    setError("");
    setLoading(true);

    const controller = new AbortController();
    activeRequest.current = controller;

    try {
      const response = await fetch(
        `${API_URL}/api/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(token
              ? {
                  Authorization: `Bearer ${token}`,
                }
              : {}),
          },
          body: JSON.stringify({
            message: question,
          }),
          signal: controller.signal,
        }
      );

      if (controller.signal.aborted) return;

      if (generation !== chatGenerationRef.current) {
        return;
      }

      // Handle expired or invalid conversation.
      if (
        response.status === 401 ||
        response.status === 410
      ) {
        updateAccessToken(null);

        setError(
          "Your previous conversation has expired. " +
            "Click New Chat to start a fresh conversation."
        );

        return;
      }

      if (!response.ok) {
        throw new Error(
          `Unable to get a response (HTTP ${response.status}).`
        );
      }

      const data = await response.json();

      if (controller.signal.aborted) return;

      if (generation !== chatGenerationRef.current) {
        return;
      }

      // Save the new signed conversation token.
      if (data.access_token) {
        updateAccessToken(data.access_token);
      }

      // Display the AI answer.
      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: data.answer,
          provider: data.provider,
          sources: data.sources || [],
        },
      ]);
    } catch (err) {
      if (err.name !== "AbortError") {
        console.error("Chat request failed:", err);

        setError(
          err.message ||
            "Connection failed. Please try again."
        );
      }
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null;
        loadingRef.current = false;
        setLoading(false);
      }
    }
  }

  // ==========================================
  // NEW CHAT
  // ==========================================

  function startNewChat() {
    // Ignore any responses belonging to
    // the previous conversation.
    chatGenerationRef.current += 1;

    activeRequest.current?.abort();
    activeRequest.current = null;

    updateAccessToken(null);

    setMessages([]);
    setInput("");
    setError("");
    setLoading(false);
    setRestoringHistory(false);
    loadingRef.current = false;

    inputRef.current?.focus();
  }

  // ==========================================
  // FORM SUBMISSION
  // ==========================================

  function handleSubmit(event) {
    event.preventDefault();
    sendMessage(input);
  }

  // ==========================================
  // RENDER
  // ==========================================

  return (
    <div className="app-shell">
      {/* =====================================
          HEADER
      ===================================== */}

      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
  <img
    src="/digital-twin-icon.png"
    alt="Digital Twin AI Logo"
  />
</div>

          <div>
            <h1>Digital Twin AI</h1>
            <p>Muhammad Bilawal Khan</p>
          </div>
        </div>

        <button
          className="new-chat-button"
          onClick={startNewChat}
          type="button"
        >
          + New Chat
        </button>
      </header>

      {/* =====================================
          CHAT AREA
      ===================================== */}

      <main className="chat-area">
        {restoringHistory ? (
          // ==================================
          // RESTORING HISTORY
          // ==================================

          <section className="welcome">
            <div className="welcome-icon">✦</div>

            <h2>Restoring your conversation...</h2>

            <p>
              Loading your previous messages.
            </p>

            <div className="typing-indicator restore-indicator">
              <span />
              <span />
              <span />
            </div>
          </section>
        ) : messages.length === 0 ? (
          // ==================================
          // WELCOME SCREEN
          // ==================================

          <section className="welcome">
            <div className="welcome-icon">✦</div>

            <div className="eyebrow">
              AI-powered professional portfolio
            </div>

            <h2>Meet My Digital Twin</h2>

            <p>
              Explore my professional experience,
              technical skills, projects, and
              achievements through an AI conversation.
            </p>

            <div className="starter-grid">
              {STARTER_QUESTIONS.map((question) => (
                <button
                  key={question}
                  className="starter-card"
                  onClick={() => sendMessage(question)}
                  disabled={loading || restoringHistory}
                  type="button"
                >
                  <span>{question}</span>

                  <span className="starter-arrow">
                    ↗
                  </span>
                </button>
              ))}
            </div>
          </section>
        ) : (
          // ==================================
          // CHAT MESSAGES
          // ==================================

          <div className="message-list">
            {messages.map((message, index) => (
              <div
                className={`message-row ${message.role}`}
                key={index}
              >
                <div className="message-avatar">
                  {message.role === "user"
                    ? "U"
                    : "✦"}
                </div>

                <div className="message-content">
                  <div className="message-author">
                    {message.role === "user"
                      ? "You"
                      : "Digital Twin AI"}
                  </div>

                  {/* ==========================
                      MESSAGE BODY
                  ========================== */}

                  <div className="message-text">
                    {message.role === "assistant" ? (
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          table: ({ children }) => (
                            <div className="markdown-table-wrapper">
                              <table>{children}</table>
                            </div>
                          ),

                          a: ({ children, href }) => (
                            <a
                              href={href}
                              target="_blank"
                              rel="noopener noreferrer"
                            >
                              {children}
                            </a>
                          ),
                        }}
                      >
                        {message.content}
                      </ReactMarkdown>
                    ) : (
                      message.content
                    )}
                  </div>

                  {/* ==========================
                      AI RESPONSE DETAILS
                  ========================== */}

                  {message.role === "assistant" && (
                    <div className="message-details">
                      {message.provider && (
                        <span className="provider-label">
                          Powered by {message.provider}
                        </span>
                      )}

                      {message.sources?.length > 0 && (
                        <details className="sources">
                          <summary>
                            View sources (
                            {message.sources.length})
                          </summary>

                          <ul>
                            {message.sources.map(
                              (source, sourceIndex) => (
                                <li key={sourceIndex}>
                                  {source.section}
                                </li>
                              )
                            )}
                          </ul>
                        </details>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* ==========================
                TYPING INDICATOR
            ========================== */}

            {loading && (
              <div className="message-row assistant">
                <div className="message-avatar">
                  ✦
                </div>

                <div className="message-content">
                  <div className="message-author">
                    Digital Twin AI
                  </div>

                  <div className="typing-indicator">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        )}
      </main>

      {/* =====================================
          MESSAGE COMPOSER
      ===================================== */}

      <footer className="composer-area">
        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        <form
          className="composer"
          onSubmit={handleSubmit}
        >
          <textarea
            ref={inputRef}
            value={input}
            onChange={(event) =>
              setInput(event.target.value)
            }
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                !event.shiftKey
              ) {
                event.preventDefault();

                if (!loading && !restoringHistory) {
                  sendMessage(input);
                }
              }
            }}
            placeholder={
              restoringHistory
                ? "Restoring conversation..."
                : "Ask me anything about my experience..."
            }
            rows={1}
            maxLength={2000}
            disabled={loading || restoringHistory}
          />

          <button
            className="send-button"
            type="submit"
            disabled={
              !input.trim() ||
              loading ||
              restoringHistory
            }
            aria-label="Send message"
          >
            ↑
          </button>
        </form>

        <p className="disclaimer">
          AI-generated responses based on documented
          professional information.
        </p>
      </footer>
    </div>
  );
}

export default App;

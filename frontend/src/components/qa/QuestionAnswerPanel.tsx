"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Sparkles,
  AlertCircle,
  HelpCircle,
  ArrowRight,
  RefreshCw,
  MessageSquare,
  SlidersHorizontal,
  FileQuestion,
  ShieldCheck,
  ShieldAlert,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Citation, GenerationResult, QAMessage, AnswerStyle } from "@/types";
import { apiClient } from "@/lib/api";
import { EvidenceCard } from "@/components/evidence/EvidenceCard";
import { EvidenceInspector } from "@/components/evidence/EvidenceInspector";
import {
  parseInlineCitations,
  getCitationLabel,
  getValidationStatus,
} from "@/lib/citationUtils";
import { PipelineTimeline } from "./PipelineTimeline";

interface QuestionAnswerPanelProps {
  documentId: string;
  documentName: string;
  onNavigateToPage: (pageNumber: number, snippet?: string, citation?: Citation) => void;
  activeCitationPage?: number | null;
  activeCitation?: Citation | null;
  onSelectCitation?: (citation: Citation | null) => void;
  onTraceUpdate?: (trace: any, citations: Citation[]) => void;
  isSideProcessActive?: boolean;
  onToggleSideProcess?: () => void;
}

const SUGGESTED_QUESTIONS = [
  "What methodology did the authors use?",
  "Which model performed best according to the benchmarks?",
  "What are the main findings and conclusions?",
  "What limitations or future work are identified?",
];

interface FormattedMarkdownAnswerProps {
  answer: string;
  citations?: Citation[];
  activeCitation?: Citation | null;
  onSelectCitation: (citation: Citation) => void;
}

const FormattedMarkdownAnswer: React.FC<FormattedMarkdownAnswerProps> = ({
  answer,
  citations = [],
  activeCitation,
  onSelectCitation,
}) => {
  const renderInline = (text: string, keyPrefix: string) => {
    const segments = parseInlineCitations(text, citations);

    return segments.map((seg, sIdx) => {
      const segKey = `${keyPrefix}_s_${sIdx}`;
      if (seg.type === "citation") {
        const cit = seg.citation;
        if (!cit) {
          return (
            <span
              key={segKey}
              className="inline-block mx-0.5 px-1.5 py-0.5 rounded text-[11px] font-mono bg-surface-low text-text-tertiary border border-border-subtle"
            >
              {seg.content}
            </span>
          );
        }
        const isSelected =
          activeCitation?.chunk_id === cit.chunk_id ||
          (activeCitation?.rank === cit.rank && activeCitation?.page_number === cit.page_number);

        return (
          <button
            key={segKey}
            type="button"
            onClick={() => onSelectCitation(cit)}
            className={`inline-flex items-center gap-1 mx-1 px-1.5 py-0.5 rounded text-[11px] font-medium transition-all ${
              isSelected
                ? "bg-brand text-white shadow-xs scale-105"
                : "bg-brand-light text-brand hover:bg-brand hover:text-white border border-brand/30"
            }`}
            title={cit.evidence_text || cit.text || `Jump to Page ${cit.page_number}`}
          >
            <span>{seg.content}</span>
          </button>
        );
      }

      // Parse inline markdown tokens: ***bold-italic***, **bold**, *italic*, `code`
      const INLINE_REGEX = /(\*\*\*[^*]+\*\*\*|\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
      const parts = seg.content.split(INLINE_REGEX);

      return (
        <span key={segKey}>
          {parts.map((part, pIdx) => {
            const partKey = `${segKey}_p_${pIdx}`;
            if (!part) return null;
            if (part.startsWith("***") && part.endsWith("***") && part.length >= 6) {
              return (
                <strong key={partKey} className="font-semibold text-text-primary">
                  <em>{part.slice(3, -3)}</em>
                </strong>
              );
            }
            if (part.startsWith("**") && part.endsWith("**") && part.length >= 4) {
              return (
                <strong key={partKey} className="font-semibold text-text-primary">
                  {part.slice(2, -2)}
                </strong>
              );
            }
            if (part.startsWith("*") && part.endsWith("*") && part.length >= 2) {
              return (
                <em key={partKey} className="italic text-text-primary">
                  {part.slice(1, -1)}
                </em>
              );
            }
            if (part.startsWith("`") && part.endsWith("`") && part.length >= 2) {
              return (
                <code
                  key={partKey}
                  className="px-1 py-0.5 rounded bg-surface-container border border-border-subtle font-mono text-[11px] text-text-primary"
                >
                  {part.slice(1, -1)}
                </code>
              );
            }
            return <span key={partKey}>{part}</span>;
          })}
        </span>
      );
    });
  };

  const rawLines = (answer || "").split(/\r?\n/);
  type Block =
    | { type: "h1" | "h2" | "h3"; text: string }
    | { type: "hr" }
    | { type: "ul"; items: string[] }
    | { type: "ol"; items: string[] }
    | { type: "p"; lines: string[] };

  const blocks: Block[] = [];

  for (const line of rawLines) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }

    if (/^(\*\*\*|---|___)$/.test(trimmed)) {
      blocks.push({ type: "hr" });
      continue;
    }

    if (trimmed.startsWith("### ")) {
      blocks.push({ type: "h3", text: trimmed.slice(4).trim() });
      continue;
    }
    if (trimmed.startsWith("## ")) {
      blocks.push({ type: "h2", text: trimmed.slice(3).trim() });
      continue;
    }
    if (trimmed.startsWith("# ")) {
      blocks.push({ type: "h1", text: trimmed.slice(2).trim() });
      continue;
    }

    if (/^[*\-+]\s+/.test(trimmed)) {
      const itemContent = trimmed.replace(/^[*\-+]\s+/, "");
      const lastBlock = blocks[blocks.length - 1];
      if (lastBlock && lastBlock.type === "ul") {
        lastBlock.items.push(itemContent);
      } else {
        blocks.push({ type: "ul", items: [itemContent] });
      }
      continue;
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      const itemContent = trimmed.replace(/^\d+\.\s+/, "");
      const lastBlock = blocks[blocks.length - 1];
      if (lastBlock && lastBlock.type === "ol") {
        lastBlock.items.push(itemContent);
      } else {
        blocks.push({ type: "ol", items: [itemContent] });
      }
      continue;
    }

    const lastBlock = blocks[blocks.length - 1];
    if (lastBlock && lastBlock.type === "p") {
      lastBlock.lines.push(trimmed);
    } else {
      blocks.push({ type: "p", lines: [trimmed] });
    }
  }

  return (
    <div className="text-xs sm:text-sm text-text-primary leading-relaxed space-y-2">
      {blocks.map((block, bIdx) => {
        const bKey = `b_${bIdx}`;
        switch (block.type) {
          case "h1":
            return (
              <h1 key={bKey} className="font-heading font-bold text-base text-text-primary pt-2 pb-0.5 border-b border-border-subtle">
                {renderInline(block.text, bKey)}
              </h1>
            );
          case "h2":
            return (
              <h2 key={bKey} className="font-heading font-bold text-sm sm:text-base text-text-primary pt-1.5 pb-0.5">
                {renderInline(block.text, bKey)}
              </h2>
            );
          case "h3":
            return (
              <h3 key={bKey} className="font-heading font-semibold text-xs sm:text-sm text-text-primary pt-1">
                {renderInline(block.text, bKey)}
              </h3>
            );
          case "hr":
            return <hr key={bKey} className="my-2 border-border-subtle" />;
          case "ul":
            return (
              <ul key={bKey} className="my-1.5 space-y-1 list-disc list-outside pl-4 text-text-primary">
                {block.items.map((item, iIdx) => (
                  <li key={`${bKey}_li_${iIdx}`} className="leading-relaxed">
                    {renderInline(item, `${bKey}_li_${iIdx}`)}
                  </li>
                ))}
              </ul>
            );
          case "ol":
            return (
              <ol key={bKey} className="my-1.5 space-y-1 list-decimal list-outside pl-4 text-text-primary">
                {block.items.map((item, iIdx) => (
                  <li key={`${bKey}_oli_${iIdx}`} className="leading-relaxed">
                    {renderInline(item, `${bKey}_oli_${iIdx}`)}
                  </li>
                ))}
              </ol>
            );
          case "p":
            return (
              <p key={bKey} className="leading-relaxed my-1">
                {renderInline(block.lines.join(" "), bKey)}
              </p>
            );
        }
      })}
    </div>
  );
};

export const QuestionAnswerPanel: React.FC<QuestionAnswerPanelProps> = ({
  documentId,
  documentName,
  onNavigateToPage,
  activeCitationPage,
  activeCitation: externalActiveCitation,
  onSelectCitation,
  onTraceUpdate,
  isSideProcessActive,
  onToggleSideProcess,
}) => {
  const [questionText, setQuestionText] = useState("");
  const [messages, setMessages] = useState<QAMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [internalActiveCitation, setInternalActiveCitation] = useState<Citation | null>(null);
  const [answerStyle, setAnswerStyle] = useState<AnswerStyle>("balanced");

  useEffect(() => {
    if (typeof window !== "undefined") {
      const stored = localStorage.getItem("doculens_answer_style") as AnswerStyle;
      if (stored && ["concise", "balanced", "detailed"].includes(stored)) {
        setAnswerStyle(stored);
      }
    }
  }, []);

  const handleStyleChange = (style: AnswerStyle) => {
    setAnswerStyle(style);
    if (typeof window !== "undefined") {
      localStorage.setItem("doculens_answer_style", style);
    }
  };

  // Inspector state
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [inspectorCitations, setInspectorCitations] = useState<Citation[]>([]);
  const [inspectorSelectedCitation, setInspectorSelectedCitation] = useState<Citation | null>(null);
  const [inspectorIsGrounded, setInspectorIsGrounded] = useState(true);
  const [showPipeline, setShowPipeline] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const storedPipeline = localStorage.getItem("doculens_show_pipeline");
      if (storedPipeline !== null) {
        setShowPipeline(storedPipeline === "true");
      }
    }
  }, []);

  const togglePipeline = () => {
    setShowPipeline((prev) => {
      const next = !prev;
      if (typeof window !== "undefined") {
        localStorage.setItem("doculens_show_pipeline", String(next));
      }
      return next;
    });
  };

  const activeCitation =
    externalActiveCitation !== undefined ? externalActiveCitation : internalActiveCitation;

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSelectCitation = (cit: Citation | null) => {
    setInternalActiveCitation(cit);
    if (onSelectCitation) onSelectCitation(cit);
    if (cit) {
      onNavigateToPage(cit.page_number, cit.evidence_text || cit.text || "", cit);
    }
  };

  const handleOpenInspector = (
    citations: Citation[],
    initialCitation?: Citation,
    isGrounded = true
  ) => {
    setInspectorCitations(citations);
    setInspectorSelectedCitation(initialCitation || citations[0] || null);
    setInspectorIsGrounded(isGrounded);
    setInspectorOpen(true);
  };

  const handleAsk = async (qText: string) => {
    const trimmed = qText.trim();
    if (!trimmed || isLoading) return;

    const messageId = `msg_${Date.now()}`;
    const newMsg: QAMessage = {
      id: messageId,
      question: trimmed,
      isLoading: true,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, newMsg]);
    setQuestionText("");
    setIsLoading(true);

    try {
      const result: GenerationResult = await apiClient.askDocument(documentId, {
        question: trimmed,
        top_k: 5,
        answer_style: answerStyle,
      });

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === messageId
            ? { ...msg, isLoading: false, result }
            : msg
        )
      );

      // If citations exist, activate the top cited page automatically
      if (result.citations && result.citations.length > 0) {
        const topCitation = result.citations[0];
        handleSelectCitation(topCitation);
      }

      // Notify parent workspace of latest pipeline trace and citations
      if (result.pipeline_trace) {
        onTraceUpdate?.(result.pipeline_trace, result.citations || []);
      }
    } catch (err: unknown) {
      const errorMsg =
        (err as { message?: string })?.message ||
        "Failed to generate answer. Please verify the document is ready.";
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === messageId
            ? { ...msg, isLoading: false, error: errorMsg }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleAsk(questionText);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk(questionText);
    }
  };

  return (
    <div className="flex flex-col h-full bg-surface-elevated border border-border-subtle rounded-lg overflow-hidden shadow-xs">
      {/* Header */}
      <div className="px-4 py-3 bg-surface-elevated border-b border-border-subtle flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded bg-brand-light flex items-center justify-center text-brand">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h2 className="font-heading text-xs font-bold text-text-primary uppercase tracking-wider">
              Document Assistant
            </h2>
            <p className="text-[11px] text-text-tertiary">
              Grounded multimodal QA &amp; verified citations
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onToggleSideProcess ? onToggleSideProcess : togglePipeline}
            aria-pressed={isSideProcessActive !== undefined ? isSideProcessActive : showPipeline}
            className={`inline-flex items-center gap-1.5 px-2 py-1 rounded text-[11px] font-medium transition-colors border ${
              (isSideProcessActive !== undefined ? isSideProcessActive : showPipeline)
                ? "bg-brand text-white border-brand shadow-xs"
                : "bg-surface-low text-text-secondary border-border-subtle hover:text-text-primary hover:border-brand/40"
            }`}
            title="Toggle technical process view on the side"
          >
            <SlidersHorizontal className="w-3 h-3" />
            <span>{(isSideProcessActive !== undefined ? isSideProcessActive : showPipeline) ? "Hide Technical Process" : "Show Technical Process"}</span>
          </button>
          <span className="text-[11px] font-mono text-text-tertiary bg-surface-low px-2 py-0.5 rounded border border-border-subtle hidden sm:inline">
            Grounded Q&amp;A
          </span>
        </div>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0 bg-surface-low/30">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-4 max-w-md mx-auto space-y-4">
            <div className="w-12 h-12 rounded-full bg-brand-light flex items-center justify-center text-brand">
              <MessageSquare className="w-6 h-6" />
            </div>
            <div className="space-y-1">
              <h3 className="font-heading text-sm font-semibold text-text-primary">
                Ask anything about this document
              </h3>
              <p className="text-xs text-text-secondary leading-relaxed">
                DocuLens retrieves verified text and visual evidence chunks from the document to
                answer your questions with traceable citations.
              </p>
            </div>

            <div className="w-full pt-2 space-y-2 text-left">
              <span className="text-[11px] font-semibold text-text-tertiary uppercase tracking-wider block px-1">
                Suggested questions:
              </span>
              <div className="space-y-1.5">
                {SUGGESTED_QUESTIONS.map((q, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleAsk(q)}
                    className="w-full text-left p-2.5 rounded-lg bg-surface-elevated border border-border-subtle hover:border-brand hover:bg-brand-light/20 transition-all text-xs text-text-secondary hover:text-text-primary flex items-center justify-between group"
                  >
                    <span className="truncate pr-2">{q}</span>
                    <ArrowRight className="w-3.5 h-3.5 text-text-tertiary group-hover:text-brand group-hover:translate-x-0.5 transition-all shrink-0" />
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div key={msg.id} className="space-y-3">
              {/* User Question */}
              <div className="flex items-start justify-end gap-2">
                <div className="bg-brand text-white rounded-2xl rounded-tr-xs px-3.5 py-2.5 max-w-[85%] sm:max-w-[75%] shadow-xs">
                  <p className="text-xs sm:text-sm leading-relaxed">{msg.question}</p>
                  <span className="text-[10px] text-blue-200 block text-right mt-1 font-mono">
                    {msg.timestamp}
                  </span>
                </div>
              </div>

              {/* Assistant Answer */}
              <div className="flex items-start gap-2.5">
                <div className="w-7 h-7 rounded-full bg-brand-light border border-border-subtle flex items-center justify-center text-brand shrink-0 mt-0.5">
                  <Sparkles className="w-3.5 h-3.5" />
                </div>

                <div className="flex-1 min-w-0 bg-surface-elevated border border-border-subtle rounded-2xl rounded-tl-xs p-3.5 sm:p-4 shadow-xs space-y-3">
                  {msg.isLoading ? (
                    <div className="flex items-center gap-2 text-xs text-text-secondary py-2" aria-live="polite">
                      <RefreshCw className="w-4 h-4 animate-spin text-brand" />
                      <span>Retrieving multimodal evidence &amp; generating grounded answer...</span>
                    </div>
                  ) : msg.error ? (
                    <div className="flex items-start gap-2 text-xs text-status-failed bg-rose-50 p-3 rounded-lg border border-rose-200" role="alert">
                      <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                      <div className="space-y-1">
                        <p className="font-semibold">Generation Error</p>
                        <p className="text-[11px] text-rose-700">{msg.error}</p>
                      </div>
                    </div>
                  ) : msg.result ? (
                    <div className="space-y-3">
                      {/* Grounding Notice if ungrounded */}
                      {!msg.result.is_grounded && (
                        <div className="flex items-center gap-2 p-2.5 bg-amber-50 rounded-lg border border-amber-300 text-xs text-amber-900">
                          <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0" />
                          <span className="text-[11px] font-medium">
                            Needs Review: Answer could not be completely grounded in retrieved evidence.
                          </span>
                        </div>
                      )}

                      {/* Answer Text with Interactive Inline Citations and Formatted Markdown */}
                      <FormattedMarkdownAnswer
                        answer={msg.result.answer}
                        citations={msg.result.citations}
                        activeCitation={activeCitation}
                        onSelectCitation={handleSelectCitation}
                      />

                      {/* Supporting Evidence Cards Section */}
                      {msg.result.citations && msg.result.citations.length > 0 ? (
                        <div className="pt-2 border-t border-border-subtle/80 space-y-2.5">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-1.5">
                              <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
                              <span className="text-xs font-semibold text-text-primary">
                                Supporting Evidence ({msg.result.citations.length})
                              </span>
                            </div>
                            <button
                              type="button"
                              onClick={() =>
                                handleOpenInspector(
                                  msg.result!.citations,
                                  undefined,
                                  msg.result!.is_grounded
                                )
                              }
                              className="inline-flex items-center gap-1 text-[11px] font-medium text-brand hover:text-brand-dark transition-colors px-1.5 py-0.5 rounded hover:bg-brand-light"
                            >
                              <SlidersHorizontal className="w-3 h-3" />
                              <span>Inspect All Evidence</span>
                            </button>
                          </div>

                          <div className="space-y-2">
                            {msg.result.citations.map((cit, idx) => {
                              const isSelected =
                                activeCitation?.chunk_id === cit.chunk_id ||
                                (activeCitation?.rank === cit.rank &&
                                  activeCitation?.page_number === cit.page_number);

                              return (
                                <EvidenceCard
                                  key={`ev_${cit.chunk_id || idx}`}
                                  citation={cit}
                                  documentName={documentName}
                                  isGrounded={msg.result!.is_grounded}
                                  isSelected={isSelected}
                                  onNavigateToPage={(p, s, c) => handleSelectCitation(c || cit)}
                                  onInspect={(c) =>
                                    handleOpenInspector(
                                      msg.result!.citations,
                                      c,
                                      msg.result!.is_grounded
                                    )
                                  }
                                />
                              );
                            })}
                          </div>
                        </div>
                      ) : (
                        <div className="pt-2 border-t border-border-subtle flex items-center gap-1.5 text-xs text-text-tertiary">
                          <FileQuestion className="w-3.5 h-3.5" />
                          <span>No supporting evidence citations were returned.</span>
                        </div>
                      )}

                      {/* Pipeline Trace Toggle */}
                      {msg.result.pipeline_trace && (
                        <div className="mt-3 pt-2 border-t border-border-subtle">
                          <button
                            type="button"
                            onClick={() => setShowPipeline(!showPipeline)}
                            className="text-[11px] font-medium text-text-tertiary hover:text-text-primary transition-colors flex items-center gap-1"
                          >
                            <SlidersHorizontal className="w-3 h-3" />
                            {showPipeline ? "Hide Technical Process" : "Show Technical Process"}
                          </button>
                          
                          {showPipeline && (
                            <PipelineTimeline 
                              trace={msg.result.pipeline_trace} 
                              citations={msg.result.citations}
                              onNavigateToPage={(p, s, c) => handleSelectCitation(c || null)} 
                              onInspectEvidence={(c) => handleOpenInspector(msg.result?.citations || [], c, msg.result?.is_grounded)}
                            />
                          )}
                        </div>
                      )}
                    </div>
                  ) : null}
                </div>
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-3 sm:p-4 bg-surface-elevated border-t border-border-subtle shrink-0">
        <form onSubmit={handleSubmit} className="space-y-2">
          {/* Style Selector Controls */}
          <div className="flex items-center justify-between px-1">
            <div className="flex items-center gap-1.5 text-text-tertiary">
              <SlidersHorizontal className="w-3.5 h-3.5" />
              <span className="text-[11px] font-medium text-text-secondary">Style:</span>
            </div>
            <div className="inline-flex items-center gap-1 bg-surface-low p-0.5 rounded-md border border-border-subtle" role="radiogroup" aria-label="Answer style">
              {(["concise", "balanced", "detailed"] as AnswerStyle[]).map((st) => (
                <button
                  key={st}
                  type="button"
                  role="radio"
                  aria-checked={answerStyle === st}
                  onClick={() => handleStyleChange(st)}
                  className={`px-2 py-0.5 rounded text-[11px] font-medium transition-all capitalize ${
                    answerStyle === st
                      ? "bg-brand text-white shadow-xs"
                      : "text-text-secondary hover:text-text-primary hover:bg-surface-elevated"
                  }`}
                  title={
                    st === "concise"
                      ? "Concise: short direct facts, 1-2 sentences"
                      : st === "balanced"
                      ? "Balanced: conversational explanation with context"
                      : "Detailed: comprehensive structured analysis with headings and bullets"
                  }
                >
                  {st}
                </button>
              ))}
            </div>
          </div>

          <div className="relative flex items-end gap-2 bg-surface-low border border-border-subtle rounded-lg p-2 focus-within:border-brand focus-within:ring-1 focus-within:ring-brand transition-all">
            <textarea
              ref={textareaRef}
              rows={2}
              value={questionText}
              onChange={(e) => setQuestionText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything about this document..."
              disabled={isLoading}
              className="flex-1 bg-transparent border-0 resize-none text-xs sm:text-sm text-text-primary placeholder:text-text-tertiary focus:outline-none disabled:opacity-50 min-h-[38px] max-h-32"
              aria-label="Ask a question about this document"
            />
            <Button
              type="submit"
              variant="primary"
              size="sm"
              disabled={!questionText.trim() || isLoading}
              className="h-8 px-3 shrink-0 rounded-md"
              aria-label="Send question"
            >
              {isLoading ? (
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Send className="w-3.5 h-3.5" />
              )}
            </Button>
          </div>
          <div className="flex items-center justify-between text-[10px] text-text-tertiary px-1">
            <span>
              Press{" "}
              <kbd className="px-1 py-0.5 bg-surface-container rounded border border-border-subtle font-mono text-[9px]">
                Enter
              </kbd>{" "}
              to submit,{" "}
              <kbd className="px-1 py-0.5 bg-surface-container rounded border border-border-subtle font-mono text-[9px]">
                Shift+Enter
              </kbd>{" "}
              for newline
            </span>
            <span>Grounded Q&amp;A</span>
          </div>
        </form>
      </div>

      {/* Evidence Inspector Modal */}
      <EvidenceInspector
        isOpen={inspectorOpen}
        onClose={() => setInspectorOpen(false)}
        citations={inspectorCitations}
        initialCitation={inspectorSelectedCitation}
        documentName={documentName}
        isGrounded={inspectorIsGrounded}
        onNavigateToPage={(page, snippet, cit) => handleSelectCitation(cit || null)}
      />
    </div>
  );
};

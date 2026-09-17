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
import { Citation, GenerationResult, QAMessage } from "@/types";
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
}

const SUGGESTED_QUESTIONS = [
  "What methodology did the authors use?",
  "Which model performed best according to the benchmarks?",
  "What are the main findings and conclusions?",
  "What limitations or future work are identified?",
];

export const QuestionAnswerPanel: React.FC<QuestionAnswerPanelProps> = ({
  documentId,
  documentName,
  onNavigateToPage,
  activeCitationPage,
  activeCitation: externalActiveCitation,
  onSelectCitation,
}) => {
  const [questionText, setQuestionText] = useState("");
  const [messages, setMessages] = useState<QAMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [internalActiveCitation, setInternalActiveCitation] = useState<Citation | null>(null);

  // Inspector state
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [inspectorCitations, setInspectorCitations] = useState<Citation[]>([]);
  const [inspectorSelectedCitation, setInspectorSelectedCitation] = useState<Citation | null>(null);
  const [inspectorIsGrounded, setInspectorIsGrounded] = useState(true);
  const [showPipeline, setShowPipeline] = useState(false);

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
        <span className="text-[11px] font-mono text-text-tertiary bg-surface-low px-2 py-0.5 rounded border border-border-subtle hidden sm:inline">
          Grounded Q&amp;A
        </span>
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

                      {/* Answer Text with Interactive Inline Citations */}
                      <div className="text-xs sm:text-sm text-text-primary leading-relaxed whitespace-pre-wrap">
                        {parseInlineCitations(msg.result.answer, msg.result.citations).map(
                          (seg, sIdx) => {
                            if (seg.type === "text") {
                              return <span key={`txt_${sIdx}`}>{seg.content}</span>;
                            }
                            const cit = seg.citation;
                            if (!cit) {
                              return (
                                <span
                                  key={`cit_${sIdx}`}
                                  className="inline-block mx-0.5 px-1.5 py-0.2 rounded text-[11px] font-mono bg-surface-low text-text-tertiary border border-border-subtle"
                                >
                                  {seg.content}
                                </span>
                              );
                            }
                            const isSelected =
                              activeCitation?.chunk_id === cit.chunk_id ||
                              (activeCitation?.rank === cit.rank &&
                                activeCitation?.page_number === cit.page_number);

                            return (
                              <button
                                key={`cit_${sIdx}`}
                                type="button"
                                onClick={() => handleSelectCitation(cit)}
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
                        )}
                      </div>

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
                            {showPipeline ? "Hide pipeline" : "Show pipeline"}
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

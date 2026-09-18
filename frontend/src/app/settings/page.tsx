"use client";

import React, { useState } from "react";
import { SlidersHorizontal, ChevronDown, Check } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";

type AnswerStyle = "concise" | "balanced" | "detailed";

const STYLES: { value: AnswerStyle; label: string; desc: string }[] = [
  { value: "concise", label: "Concise", desc: "Direct facts" },
  { value: "balanced", label: "Balanced", desc: "Thorough context" },
  { value: "detailed", label: "Detailed", desc: "In-depth" },
];

export default function SettingsPage() {
  const [answerStyle, setAnswerStyle] = useState<AnswerStyle>("balanced");
  const [showSources, setShowSources] = useState(true);
  const [highlight, setHighlight] = useState(true);
  const [showPipeline, setShowPipeline] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [saved, setSaved] = useState(false);

  React.useEffect(() => {
    if (typeof window !== "undefined") {
      const stored = localStorage.getItem("doculens_answer_style") as AnswerStyle;
      if (stored && ["concise", "balanced", "detailed"].includes(stored)) {
        setAnswerStyle(stored);
      }
      const storedPipeline = localStorage.getItem("doculens_show_pipeline");
      if (storedPipeline !== null) {
        setShowPipeline(storedPipeline === "true");
      }
    }
  }, []);

  const handleStyleChange = (val: AnswerStyle) => {
    setAnswerStyle(val);
    if (typeof window !== "undefined") {
      localStorage.setItem("doculens_answer_style", val);
    }
  };

  const handleTogglePipeline = (val: boolean) => {
    setShowPipeline(val);
    if (typeof window !== "undefined") {
      localStorage.setItem("doculens_show_pipeline", String(val));
    }
  };

  const handleSave = () => {
    if (typeof window !== "undefined") {
      localStorage.setItem("doculens_answer_style", answerStyle);
      localStorage.setItem("doculens_show_pipeline", String(showPipeline));
    }
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="space-y-6 max-w-3xl">
      <div>
        <h1 className="font-heading text-2xl font-bold tracking-tight text-text-primary">Settings</h1>
        <p className="text-sm text-text-secondary mt-0.5">Customize your DocuLens AI experience.</p>
      </div>

      {/* Answer Preferences */}
      <Card header={<h2 className="font-heading text-sm font-semibold text-text-primary uppercase tracking-wide">Answer Preferences</h2>}>
        <div className="space-y-5">
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Answer Style</label>
            <div className="grid grid-cols-3 gap-2">
              {STYLES.map(({ value, label, desc }) => (
                <button key={value} type="button" onClick={() => handleStyleChange(value)}
                  className={`p-2.5 text-center rounded border ${
                    answerStyle === value ? "bg-text-primary text-white border-text-primary" : "border-border-subtle text-text-secondary bg-surface-low"
                  }`}>
                  <div className="text-xs font-semibold uppercase">{label}</div>
                  <div className="text-[10px] mt-0.5 opacity-70">{desc}</div>
                </button>
              ))}
            </div>
          </div>
          <div className="space-y-3 pt-3 border-t border-border-subtle">
            <label className="flex items-start gap-3 cursor-pointer">
              <input type="checkbox" checked={showSources} onChange={(e) => setShowSources(e.target.checked)}
                className="mt-0.5 w-4 h-4 border-border-strong text-brand focus:ring-0 cursor-pointer" />
              <div>
                <div className="text-sm font-semibold text-text-primary">Always show sources</div>
                <div className="text-xs text-text-secondary mt-0.5">Append page citations to every answer.</div>
              </div>
            </label>
            <label className="flex items-start gap-3 cursor-pointer">
              <input type="checkbox" checked={highlight} onChange={(e) => setHighlight(e.target.checked)}
                className="mt-0.5 w-4 h-4 border-border-strong text-brand focus:ring-0 cursor-pointer" />
              <div>
                <div className="text-sm font-semibold text-text-primary">Highlight evidence</div>
                <div className="text-xs text-text-secondary mt-0.5">Emphasize retrieved evidence passages.</div>
              </div>
            </label>
          </div>
        </div>
      </Card>

      {/* Document Preferences */}
      <Card header={<h2 className="font-heading text-sm font-semibold text-text-primary uppercase tracking-wide">Document Preferences</h2>}>
        <p className="text-sm text-text-secondary py-2">Upload limits and handling options will appear here in a future update.</p>
      </Card>

      {/* Advanced */}
      <Card header={
        <button type="button" onClick={() => setShowAdvanced((p) => !p)} className="w-full flex items-center justify-between">
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-text-tertiary" />
            <h2 className="font-heading text-sm font-semibold text-text-primary uppercase tracking-wide">Advanced</h2>
            <span className="text-[10px] px-1.5 py-0.5 bg-surface-container border border-border-subtle text-text-secondary rounded uppercase font-mono">Technical</span>
          </div>
          <ChevronDown className={`w-4 h-4 text-text-tertiary transition-transform ${showAdvanced ? "rotate-180" : ""}`} />
        </button>
      }>
        {showAdvanced ? (
          <div className="space-y-4 py-2">
            <label className="flex items-start gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={showPipeline}
                onChange={(e) => handleTogglePipeline(e.target.checked)}
                className="mt-0.5 w-4 h-4 border-border-strong text-brand focus:ring-0 cursor-pointer"
              />
              <div>
                <div className="text-sm font-semibold text-text-primary">Show internal pipeline trace</div>
                <div className="text-xs text-text-secondary mt-0.5">
                  Display hybrid retrieval, BM25/vector fusion, and reranking pipeline telemetry in the Q&amp;A panel.
                </div>
              </div>
            </label>
            <p className="text-xs text-text-tertiary pt-2 border-t border-border-subtle">
              Pipeline preferences: Grounded LLM generation with hybrid dense vector and BM25 lexical retrieval.
            </p>
          </div>
        ) : (
          <p className="text-xs text-text-tertiary py-2">Expand to reveal technical settings and pipeline visibility.</p>
        )}
      </Card>

      <div className="flex justify-end">
        <Button variant="primary" onClick={handleSave} leftIcon={saved ? <Check className="w-4 h-4" /> : undefined}>
          {saved ? "Preferences Saved" : "Save Preferences"}
        </Button>
      </div>
    </div>
  );
}

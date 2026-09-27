import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { BookOpen, MessageCircle, SendHorizonal, Sparkles, X } from "lucide-react";
import { LogoMark } from "@/components/brand/Logo";
import { conversation } from "@/mocks/data";
import { cn } from "@/lib/cn";

type Msg = {
  role: "user" | "assistant";
  text: string;
  citations?: { doc: string; page: number }[];
};

const cannedReply: Msg = {
  role: "assistant",
  text: "In the building phase I answer from a fixed script. Once the LLM gateway is wired, every number in my reply will be checked against your report and guideline text before you see it. Consult a doctor for medical decisions.",
  citations: [{ doc: "MediScan gateway spec", page: 1 }],
};

export function ChatWidget() {
  const [open, setOpen] = useState(false);
  const [msgs, setMsgs] = useState<Msg[]>(conversation);
  const [draft, setDraft] = useState("");
  const [typing, setTyping] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs, open, typing]);

  const send = () => {
    if (!draft.trim()) return;
    setMsgs((m) => [...m, { role: "user", text: draft.trim() }]);
    setDraft("");
    setTyping(true);
    setTimeout(() => {
      setMsgs((m) => [...m, cannedReply]);
      setTyping(false);
    }, 900);
  };

  return (
    <>
      <motion.button
        onClick={() => setOpen((o) => !o)}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        className="fixed bottom-5 right-5 z-50 grid size-14 place-items-center rounded-full bg-gradient-to-br from-brand-500 to-brand-700 text-white shadow-[0_15px_40px_-10px_rgb(16_185_129/0.7)]"
        aria-label="Open MediScan assistant"
      >
        <span className="absolute inset-0 rounded-full bg-brand-400/50 animate-pulse-ring" />
        {open ? <X className="relative size-6" /> : <MessageCircle className="relative size-6" />}
      </motion.button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 24, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 24, scale: 0.96 }}
            transition={{ type: "spring", stiffness: 300, damping: 28 }}
            className="glass-strong fixed bottom-24 right-5 z-50 flex h-[560px] w-[min(400px,calc(100vw-2.5rem))] flex-col overflow-hidden p-0"
            data-lenis-prevent
          >
            <div className="flex items-center gap-3 border-b border-line bg-white/60 px-4 py-3">
              <LogoMark size={32} />
              <div className="flex-1">
                <p className="text-sm font-bold">MediScan assistant</p>
                <p className="text-[11px] text-ink-muted">
                  Grounded in your report + clinical guidelines
                </p>
              </div>
              <span className="inline-flex items-center gap-1 rounded-full bg-brand-100 px-2 py-0.5 text-[10px] font-bold text-brand-800">
                <Sparkles className="size-3" /> RAG
              </span>
            </div>
            <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4 scrollbar-thin">
              {msgs.map((m, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={cn("flex", m.role === "user" ? "justify-end" : "justify-start")}
                >
                  <div
                    className={cn(
                      "max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed",
                      m.role === "user"
                        ? "rounded-br-md bg-gradient-to-br from-brand-500 to-brand-700 text-white"
                        : "rounded-bl-md bg-white/80 text-ink",
                    )}
                  >
                    {m.text}
                    {m.citations && (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {m.citations.map((c) => (
                          <span
                            key={c.doc}
                            className="inline-flex items-center gap-1 rounded-full bg-brand-50 px-2 py-0.5 text-[10px] font-semibold text-brand-800"
                          >
                            <BookOpen className="size-3" /> {c.doc} · p.{c.page}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
              {typing && (
                <div className="flex gap-1 rounded-2xl bg-white/80 px-4 py-3 w-fit">
                  {[0, 1, 2].map((i) => (
                    <motion.span
                      key={i}
                      className="size-1.5 rounded-full bg-brand-500"
                      animate={{ y: [0, -4, 0] }}
                      transition={{ repeat: Infinity, duration: 0.8, delay: i * 0.15 }}
                    />
                  ))}
                </div>
              )}
              <div ref={endRef} />
            </div>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                send();
              }}
              className="flex items-center gap-2 border-t border-line bg-white/60 p-3"
            >
              <input
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                placeholder="Ask about your report…"
                className="h-11 flex-1 rounded-full border border-white/80 bg-white px-4 text-sm ring-focus"
              />
              <button
                type="submit"
                className="grid size-11 place-items-center rounded-full bg-brand-600 text-white transition hover:bg-brand-700"
                aria-label="Send"
              >
                <SendHorizonal className="size-4" />
              </button>
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

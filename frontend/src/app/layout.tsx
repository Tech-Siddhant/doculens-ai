import type { Metadata } from "next";
import "./globals.css";
import { AppLayout } from "@/components/layout/AppLayout";

export const metadata: Metadata = {
  title: "DocuLens AI — Multimodal Document Intelligence",
  description:
    "Grounded question answering over research and technical PDFs with verifiable page citations.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background text-text-primary antialiased">
        <AppLayout>{children}</AppLayout>
      </body>
    </html>
  );
}

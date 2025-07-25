// Core data types based on Python implementation

export interface ProcessingResult {
  status: boolean;
  message: string;
  data: any;
}

export enum ModelType {
  OPENAI = "openai",
  CLAUDE = "claude",
  GROQ = "groq",
}

export interface ModelConfig {
  type: ModelType;
  modelName: string;
  temperature?: number;
  maxTokens?: number;
  apiKey?: string;
}

export interface LlamaParseConfig {
  apiKey: string;
  resultType: "markdown" | "text";
  language: string[];
  parseMode: "parse_document_with_agent" | "simple";
  model: string;
}

export interface LlamaParseApiConfig {
  parseMode?: string;
  vendorMultimodalModelName?: string;
  structuredOutput?: boolean;
  disableOcr?: boolean;
  disableImageExtraction?: boolean;
  adaptiveLongTable?: boolean;
  annotateLinks?: boolean;
  doNotUnrollColumns?: boolean;
  htmlMakeAllElementsVisible?: boolean;
  htmlRemoveNavigationElements?: boolean;
  htmlRemoveFixedElements?: boolean;
  guessXlsxSheetName?: boolean;
  doNotCache?: boolean;
  invalidateCache?: boolean;
  outputPdfOfDocument?: boolean;
  takeScreenshot?: boolean;
  isFormattingInstruction?: boolean;
  timeout?: number;
}

export interface JobStatus {
  id: string;
  status: "PENDING" | "PROCESSING" | "SUCCESS" | "FAILED";
  progress?: number;
  error?: string;
}

export interface DocumentSummaryRequest {
  filePath: string;
  userId: string;
  language: string;
}

// Utility types for better type safety
export type ApiResponse<T> =
  | {
      success: true;
      data: T;
    }
  | {
      success: false;
      error: string;
    };

export interface Logger {
  info(message: string, ...args: any[]): void;
  warn(message: string, ...args: any[]): void;
  error(message: string, ...args: any[]): void;
  debug(message: string, ...args: any[]): void;
}

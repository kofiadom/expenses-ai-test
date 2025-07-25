export interface DocumentProcessingData {
  jobId: string;
  filePath: string;
  fileName: string;
  userId: string;
  country: string;
  icp: string;
  uploadedAt: Date;
}

export interface ExpenseLineItem {
  description: string;
  amount: string;
  quantity?: number;
  category?: string;
}

// Only one job data type needed now
export type ExpenseProcessingJobData = DocumentProcessingData;

export interface ProcessingStatus {
  jobId: string;
  status: 'waiting' | 'active' | 'completed' | 'failed' | 'delayed';
  progress: {
    fileClassification: boolean;
    dataExtraction: boolean;
    issueDetection: boolean;
    citationGeneration: boolean;
  };
  results?: {
    classification?: any;
    extraction?: any;
    compliance?: any;
    citations?: any;
  };
  error?: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface JobResult {
  success: boolean;
  data?: any;
  error?: string;
  processingTime?: number;
}

// Single queue name
export const QUEUE_NAMES = {
  EXPENSE_PROCESSING: 'expense-processing',
} as const;

// Job types for the single queue - now only one job type needed
export const JOB_TYPES = {
  PROCESS_DOCUMENT: 'process-document',
} as const;

export interface ProcessingMetrics {
  totalJobs: number;
  completedJobs: number;
  failedJobs: number;
  averageProcessingTime: number;
  queueHealth: {
    [key: string]: {
      waiting: number;
      active: number;
      completed: number;
      failed: number;
    };
  };
}

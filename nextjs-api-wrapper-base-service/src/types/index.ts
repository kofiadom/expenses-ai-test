export interface DocumentProcessingData {
  jobId: string;
  filePath: string;
  fileName: string;
  userId: string;
  language: string;
  uploadedAt: Date;
}

export interface LabParameterData {
  name: string;
  result?: string;
  units?: string;
  range?: string;
  comment?: string;
}

// Only one job data type needed now
export type MedicalProcessingJobData = DocumentProcessingData;

export interface ProcessingStatus {
  jobId: string;
  status: 'waiting' | 'active' | 'completed' | 'failed' | 'delayed';
  progress: {
    documentSummary: boolean;
    physicianMatching: boolean;
    facilityMatching: boolean;
    labParameterMatching: {
      total: number;
      completed: number;
      percentage: number;
    };
  };
  results?: {
    summary?: any;
    physicianMatch?: any;
    facilityMatch?: any;
    labMatches?: any[];
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
  MEDICAL_PROCESSING: 'medical-processing',
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

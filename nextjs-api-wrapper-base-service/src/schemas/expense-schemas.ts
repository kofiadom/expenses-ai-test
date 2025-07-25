import { z } from 'zod';

// File Classification Schemas
export const SchemaFieldAnalysisSchema = z.object({
  fields_found: z.array(z.string()),
  fields_missing: z.array(z.string()),
  total_fields_found: z.number(),
  expense_identification_reasoning: z.string(),
});

export const FileClassificationResultSchema = z.object({
  is_expense: z.boolean(),
  expense_type: z.string().nullable(),
  language: z.string(),
  language_confidence: z.number(),
  document_location: z.string(),
  expected_location: z.string(),
  location_match: z.boolean(),
  error_type: z.string().nullable(),
  error_message: z.string().nullable(),
  classification_confidence: z.number(),
  reasoning: z.string(),
  schema_field_analysis: SchemaFieldAnalysisSchema,
});

// Citation Schemas
export const CitationInfoSchema = z.object({
  source_text: z.string().nullable(),
  confidence: z.number(),
  source_location: z.string().nullable(),
  context: z.string().nullable(),
  match_type: z.string(),
});

export const FieldCitationSchema = z.object({
  field_citation: CitationInfoSchema.nullable(),
  value_citation: CitationInfoSchema.nullable(),
}).strict();

export const CitationMetadataSchema = z.object({
  total_fields_analyzed: z.number(),
  fields_with_field_citations: z.number(),
  fields_with_value_citations: z.number(),
  average_confidence: z.number(),
});

export const CitationResultSchema = z.object({
  citations: z.record(z.string(), FieldCitationSchema),
  metadata: CitationMetadataSchema,
}).strict();

// Issue Detection Schemas
export const ComplianceIssueSchema = z.object({
  issue_type: z.string(),
  field: z.string(),
  description: z.string(),
  recommendation: z.string(),
  knowledge_base_reference: z.string(), // Reverted back to correct name
});

export const ValidationResultSchema = z.object({
  is_valid: z.boolean(),
  issues_count: z.number(),
  issues: z.array(ComplianceIssueSchema),
  corrected_receipt: z.null(),
  compliance_summary: z.string(),
});

export const TechnicalDetailsSchema = z.object({
  content_type: z.string(),
  country: z.string(),
  icp: z.string(),
  receipt_type: z.string(),
  issues_count: z.number(),
});

export const IssueDetectionResultSchema = z.object({
  validation_result: ValidationResultSchema,
  technical_details: TechnicalDetailsSchema,
});

// Data Extraction Schema - Flexible schema to handle any extracted data
export const ExpenseDataSchema = z.record(z.string(), z.any());

// Image Quality Assessment Schemas
export const QualityIssueSchema = z.object({
  detected: z.boolean(),
  severity_level: z.enum(['low', 'medium', 'high', 'critical']),
  confidence_score: z.number().min(0).max(1),
  quantitative_measure: z.number(),
  description: z.string(),
  recommendation: z.string(),
});

export const ImageQualityAssessmentSchema = z.object({
  blur_detection: QualityIssueSchema,
  contrast_assessment: QualityIssueSchema,
  glare_identification: QualityIssueSchema,
  water_stains: QualityIssueSchema,
  tears_or_folds: QualityIssueSchema,
  cut_off_detection: QualityIssueSchema,
  missing_sections: QualityIssueSchema,
  obstructions: QualityIssueSchema,
  overall_quality_score: z.number().min(1).max(10),
  suitable_for_extraction: z.boolean(),
});

// Processing Timing Schema
export const ProcessingTimingSchema = z.object({
  total_processing_time_minutes: z.string(),
  phase_timings: z.object({
    image_quality_assessment_minutes: z.string(),
    file_classification_minutes: z.string(),
    data_extraction_minutes: z.string(),
    issue_detection_minutes: z.string(),
    citation_generation_minutes: z.string(),
  }),
  agent_performance: z.object({
    image_quality_assessment: z.object({
      start_time: z.string(),
      end_time: z.string(),
      duration_minutes: z.string(),
      model_used: z.string(),
    }),
    file_classification: z.object({
      start_time: z.string(),
      end_time: z.string(),
      duration_minutes: z.string(),
      model_used: z.string(),
    }),
    data_extraction: z.object({
      start_time: z.string(),
      end_time: z.string(),
      duration_minutes: z.string(),
      model_used: z.string(),
    }),
    issue_detection: z.object({
      start_time: z.string(),
      end_time: z.string(),
      duration_minutes: z.string(),
      model_used: z.string(),
    }),
    citation_generation: z.object({
      start_time: z.string(),
      end_time: z.string(),
      duration_minutes: z.string(),
      model_used: z.string(),
    }),
  }),
});

// Complete Processing Result Schema
export const CompleteProcessingResultSchema = z.object({
  image_quality_assessment: z.any().nullable(), // Flexible schema for quality assessment
  classification: FileClassificationResultSchema,
  extraction: ExpenseDataSchema,
  compliance: IssueDetectionResultSchema,
  citations: CitationResultSchema,
  timing: ProcessingTimingSchema,
  metadata: z.object({
    filename: z.string(),
    processing_time: z.number(),
    country: z.string(),
    icp: z.string(),
    processed_at: z.string(),
  }),
});

// Type exports for TypeScript
export type SchemaFieldAnalysis = z.infer<typeof SchemaFieldAnalysisSchema>;
export type FileClassificationResult = z.infer<typeof FileClassificationResultSchema>;
export type CitationInfo = z.infer<typeof CitationInfoSchema>;
export type FieldCitation = z.infer<typeof FieldCitationSchema>;
export type CitationMetadata = z.infer<typeof CitationMetadataSchema>;
export type CitationResult = z.infer<typeof CitationResultSchema>;
export type ComplianceIssue = z.infer<typeof ComplianceIssueSchema>;
export type ValidationResult = z.infer<typeof ValidationResultSchema>;
export type TechnicalDetails = z.infer<typeof TechnicalDetailsSchema>;
export type IssueDetectionResult = z.infer<typeof IssueDetectionResultSchema>;
export type ExpenseData = z.infer<typeof ExpenseDataSchema>;
export type CompleteProcessingResult = z.infer<typeof CompleteProcessingResultSchema>;
export type ProcessingTiming = z.infer<typeof ProcessingTimingSchema>;
export type QualityIssue = z.infer<typeof QualityIssueSchema>;
export type ImageQualityAssessment = z.infer<typeof ImageQualityAssessmentSchema>;

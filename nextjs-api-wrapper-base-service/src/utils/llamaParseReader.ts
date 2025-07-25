import * as fs from "fs";
import { z } from "zod";
import { LlamaParseApiConfig, ApiResponse } from "./types";
import { robustFetch } from "./robustFetch";

// Zod schemas for API responses
const uploadResponseSchema = z.object({
  id: z.string(),
});

const jobStatusSchema = z.object({
  status: z.enum(["PENDING", "PROCESSING", "SUCCESS", "ERROR"]),
});

const markdownResultSchema = z.object({
  markdown: z.string(),
});

export class LlamaParseApiService {
  private apiKey: string;
  private baseUrl = "https://api.cloud.llamaindex.ai/api/parsing";
  private parseCache = new Map<string, { result: Promise<ApiResponse<string>>; timestamp: number }>();
  private cacheTimeout = 10 * 60 * 1000; // 10 minutes cache

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  /**
   * Parse document using direct API calls with full configuration options
   */
  async parseDocument(
    filePath: string,
    config: LlamaParseApiConfig = {}
  ): Promise<ApiResponse<string>> {
    // Create cache key based on file path and config
    const cacheKey = `${filePath}_${JSON.stringify(config)}`;
    const now = Date.now();

    // Clean expired cache entries
    for (const [key, entry] of this.parseCache.entries()) {
      if (now - entry.timestamp > this.cacheTimeout) {
        this.parseCache.delete(key);
      }
    }

    // Check if we have a cached result for this file
    const cachedEntry = this.parseCache.get(cacheKey);
    if (cachedEntry) {
      console.log(`Using cached result for document: ${filePath}`);
      return await cachedEntry.result;
    }

    // Create the parsing promise
    const parsePromise = this.performParsing(filePath, config);
    
    // Cache the promise immediately to prevent duplicate calls
    this.parseCache.set(cacheKey, {
      result: parsePromise,
      timestamp: now,
    });

    return await parsePromise;
  }

  /**
   * Perform the actual document parsing
   */
  private async performParsing(
    filePath: string,
    config: LlamaParseApiConfig
  ): Promise<ApiResponse<string>> {
    try {
      console.log(`Parsing document: ${filePath}`);

      // Check if it's a local file
      if (!filePath.startsWith("http") && !fs.existsSync(filePath)) {
        return {
          success: false,
          error: `File not found: ${filePath}`,
        };
      }

      // Step 1: Upload file
      const uploadResult: any = await this.uploadFile(filePath, config);
      if (!uploadResult.success) {
        return {
          success: false,
          error: uploadResult.error,
        };
      }

      const jobId = uploadResult.data.id;
      console.log(`Upload successful. Job ID: ${jobId}`);

      // Step 2: Poll for completion
      const pollResult: any = await this.pollJobCompletion(jobId, config.timeout || 300000);
      if (!pollResult.success) {
        return {
          success: false,
          error: pollResult.error,
        };
      }

      // Step 3: Get markdown result
      const markdownResult: any = await this.getJobResult(jobId);
      if (!markdownResult.success) {
        return {
          success: false,
          error: markdownResult.error,
        };
      }

      console.log(
        `Document parsed successfully. Content length: ${markdownResult.data.length} characters`
      );

      return {
        success: true,
        data: markdownResult.data,
      };
    } catch (error) {
      console.error("Error parsing document:", error);
      return {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error occurred",
      };
    }
  }

  /**
   * Upload file to LlamaParse API
   */
  private async uploadFile(
    filePath: string,
    config: LlamaParseApiConfig
  ): Promise<ApiResponse<{ id: string }>> {
    try {
      const formData = new FormData();
      
      // Read file and create blob
      const fileBuffer = fs.readFileSync(filePath);
      const blob = new Blob([fileBuffer], { type: "application/pdf" });
      formData.append("file", blob, filePath);

      // Add configuration parameters
      formData.append("parse_mode", config.parseMode || "parse_page_with_lvm");
      formData.append("vendor_multimodal_model_name", config.vendorMultimodalModelName || "anthropic-sonnet-3.7");
      formData.append("structured_output", String(config.structuredOutput || false));
      formData.append("disable_ocr", String(config.disableOcr || false));
      formData.append("disable_image_extraction", String(config.disableImageExtraction || false));
      formData.append("adaptive_long_table", String(config.adaptiveLongTable || false));
      formData.append("annotate_links", String(config.annotateLinks || false));
      formData.append("do_not_unroll_columns", String(config.doNotUnrollColumns || false));
      formData.append("html_make_all_elements_visible", String(config.htmlMakeAllElementsVisible || false));
      formData.append("html_remove_navigation_elements", String(config.htmlRemoveNavigationElements || false));
      formData.append("html_remove_fixed_elements", String(config.htmlRemoveFixedElements || false));
      formData.append("guess_xlsx_sheet_name", String(config.guessXlsxSheetName || false));
      formData.append("do_not_cache", String(config.doNotCache || false));
      formData.append("invalidate_cache", String(config.invalidateCache || false));
      formData.append("output_pdf_of_document", String(config.outputPdfOfDocument || false));
      formData.append("take_screenshot", String(config.takeScreenshot || false));
      formData.append("is_formatting_instruction", String(config.isFormattingInstruction || true));

      const response = await robustFetch({
        url: `${this.baseUrl}/upload`,
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
        },
        body: formData,
        schema: uploadResponseSchema,
        tryCount: 3,
        tryCooldown: 2000,
      });

      return {
        success: true,
        data: response as { id: string },
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : "Upload failed",
      };
    }
  }

  /**
   * Poll job status until completion
   */
  private async pollJobCompletion(
    jobId: string,
    timeout: number = 300000
  ): Promise<ApiResponse<void>> {
    const startTime = Date.now();
    const pollInterval = 10000; // 10 seconds

    while (Date.now() - startTime < timeout) {
      try {
        const statusResponse = await robustFetch({
          url: `${this.baseUrl}/job/${jobId}`,
          method: "GET",
          headers: {
            Authorization: `Bearer ${this.apiKey}`,
          },
          schema: jobStatusSchema,
          tryCount: 2,
          tryCooldown: 1000,
        });

        console.log(`Job ${jobId} status: ${statusResponse.status}`);

        if (statusResponse.status === "SUCCESS") {
          return { success: true, data: undefined };
        }

        if (statusResponse.status === "ERROR") {
          return {
            success: false,
            error: `Job failed with status: ${statusResponse.status}`,
          };
        }

        // Wait before next poll
        await new Promise((resolve) => setTimeout(resolve, pollInterval));
      } catch (error) {
        console.warn(`Error polling job status: ${error}`);
        await new Promise((resolve) => setTimeout(resolve, pollInterval));
      }
    }

    return {
      success: false,
      error: `Job timed out after ${timeout}ms`,
    };
  }

  /**
   * Get job result markdown
   */
  private async getJobResult(jobId: string): Promise<ApiResponse<string>> {
    try {
      const response = await robustFetch({
        url: `${this.baseUrl}/job/${jobId}/result/markdown`,
        method: "GET",
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
        },
        schema: markdownResultSchema,
        tryCount: 3,
        tryCooldown: 1000,
      });

      return {
        success: true,
        data: response.markdown,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : "Failed to get result",
      };
    }
  }
}

// Factory function for easy instantiation
export function createLlamaParseService(apiKey: string): LlamaParseApiService {
  return new LlamaParseApiService(apiKey);
}

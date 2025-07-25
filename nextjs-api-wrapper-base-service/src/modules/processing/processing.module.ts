import { Module } from "@nestjs/common";
import { BullModule } from "@nestjs/bull";
import { ExpenseProcessor } from "./processors/expense.processor";
import { ProcessingService } from "./services/processing.service";
import { ExpenseProcessingService } from "../../services/expense-processing.service";
import { QUEUE_NAMES } from "../../types";
import { DocumentModule } from "../document/document.module";

@Module({
  imports: [
    // Register single queue for all expense processing with extended timeouts
    BullModule.registerQueue({
      name: QUEUE_NAMES.EXPENSE_PROCESSING,
      defaultJobOptions: {
        removeOnComplete: 10,
        removeOnFail: 5,
        attempts: 3,
        backoff: {
          type: "exponential",
          delay: 2000,
        },
        // 10 minute timeout for all processing jobs
        timeout: 10 * 60 * 1000, // 600,000ms = 10 minutes
      },
    }),

 

    // Import Document services
    DocumentModule,
  ],
  providers: [ProcessingService, ExpenseProcessor, ExpenseProcessingService],
  exports: [ProcessingService, ExpenseProcessingService],
})
export class ProcessingModule {}

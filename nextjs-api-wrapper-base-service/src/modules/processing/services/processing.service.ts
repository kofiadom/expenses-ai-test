import { Injectable, Logger } from '@nestjs/common';
import { InjectQueue } from '@nestjs/bull';
import { Queue } from 'bull';
import { QUEUE_NAMES, JOB_TYPES } from '../../../types';

@Injectable()
export class ProcessingService {
  private readonly logger = new Logger(ProcessingService.name);

  constructor(
    @InjectQueue(QUEUE_NAMES.MEDICAL_PROCESSING)
    private medicalQueue: Queue,
  ) {}

  async getQueueStats() {
    try {
      const counts = await this.medicalQueue.getJobCounts();
      
      // Get job breakdown by type
      const allJobs = await this.medicalQueue.getJobs(['waiting', 'active', 'completed', 'failed']);
      
      const jobsByType = {
        [JOB_TYPES.PROCESS_DOCUMENT]: { waiting: 0, active: 0, completed: 0, failed: 0 },
      };

      // Count jobs by type and status
      allJobs.forEach(job => {
        const jobType = job.name;
        if (jobsByType[jobType]) {
          if (job.finishedOn && !job.failedReason) {
            jobsByType[jobType].completed++;
          } else if (job.failedReason) {
            jobsByType[jobType].failed++;
          } else if (job.processedOn) {
            jobsByType[jobType].active++;
          } else {
            jobsByType[jobType].waiting++;
          }
        }
      });

      return {
        [QUEUE_NAMES.MEDICAL_PROCESSING]: {
          total: counts,
          byJobType: jobsByType,
          waiting: counts.waiting || 0,
          active: counts.active || 0,
          completed: counts.completed || 0,
          failed: counts.failed || 0,
          delayed: counts.delayed || 0,
        },
      };
    } catch (error) {
      this.logger.error('Failed to get queue stats:', error);
      throw error;
    }
  }

  async pauseQueue() {
    try {
      await this.medicalQueue.pause();
      this.logger.log(`Medical processing queue paused`);
      return { success: true, message: `Medical processing queue paused` };
    } catch (error) {
      this.logger.error(`Failed to pause medical processing queue:`, error);
      throw error;
    }
  }

  async resumeQueue() {
    try {
      await this.medicalQueue.resume();
      this.logger.log(`Medical processing queue resumed`);
      return { success: true, message: `Medical processing queue resumed` };
    } catch (error) {
      this.logger.error(`Failed to resume medical processing queue:`, error);
      throw error;
    }
  }

  async cleanQueue(grace: number = 5000) {
    try {
      await this.medicalQueue.clean(grace, 'completed');
      await this.medicalQueue.clean(grace, 'failed');
      this.logger.log(`Medical processing queue cleaned`);
      return { success: true, message: `Medical processing queue cleaned` };
    } catch (error) {
      this.logger.error(`Failed to clean medical processing queue:`, error);
      throw error;
    }
  }

  async getJobsByType(jobType: string, status: string[] = ['waiting', 'active', 'completed', 'failed']) {
    try {
      const jobs = await this.medicalQueue.getJobs(status as any);
      const filteredJobs = jobs.filter(job => job.name === jobType);
      
      return filteredJobs.map(job => ({
        id: job.id,
        name: job.name,
        data: job.data,
        status: job.finishedOn ? (job.failedReason ? 'failed' : 'completed') : 
                job.processedOn ? 'active' : 'waiting',
        progress: job.progress(),
        createdAt: new Date(job.timestamp),
        processedAt: job.processedOn ? new Date(job.processedOn) : null,
        finishedAt: job.finishedOn ? new Date(job.finishedOn) : null,
        failedReason: job.failedReason,
        returnValue: job.returnvalue,
      }));
    } catch (error) {
      this.logger.error(`Failed to get jobs by type ${jobType}:`, error);
      throw error;
    }
  }

  async getHealthStatus() {
    try {
      const stats = await this.getQueueStats();
      const queueStats = stats[QUEUE_NAMES.MEDICAL_PROCESSING];
      
      // Check if queue has too many failed jobs
      const totalJobs = queueStats.completed + queueStats.failed;
      const failureRate = totalJobs > 0 ? queueStats.failed / totalJobs : 0;
      const isHealthy = failureRate < 0.1; // Less than 10% failure rate

      // Check job type health
      const jobTypeHealth = Object.entries(queueStats.byJobType).map(([jobType, counts]: [string, any]) => {
        const typeTotal = counts.completed + counts.failed;
        const typeFailureRate = typeTotal > 0 ? counts.failed / typeTotal : 0;
        const typeHealthy = typeFailureRate < 0.1;
        
        return {
          jobType,
          healthy: typeHealthy,
          failureRate: Math.round(typeFailureRate * 100),
          counts,
        };
      });

      const overallHealthy = isHealthy && jobTypeHealth.every(check => check.healthy);

      return {
        service: 'Processing Service',
        status: overallHealthy ? 'healthy' : 'degraded',
        timestamp: new Date().toISOString(),
        queue: {
          name: QUEUE_NAMES.MEDICAL_PROCESSING,
          healthy: isHealthy,
          failureRate: Math.round(failureRate * 100),
          counts: queueStats.total,
        },
        jobTypes: jobTypeHealth,
        stats,
      };
    } catch (error) {
      this.logger.error('Health check failed:', error);
      return {
        service: 'Processing Service',
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        error: error.message,
      };
    }
  }

  async retryFailedJobs(jobType?: string) {
    try {
      const failedJobs = await this.medicalQueue.getJobs(['failed']);
      const jobsToRetry = jobType 
        ? failedJobs.filter(job => job.name === jobType)
        : failedJobs;

      let retriedCount = 0;
      for (const job of jobsToRetry) {
        await job.retry();
        retriedCount++;
      }

      this.logger.log(`Retried ${retriedCount} failed jobs${jobType ? ` of type ${jobType}` : ''}`);
      return { 
        success: true, 
        message: `Retried ${retriedCount} failed jobs${jobType ? ` of type ${jobType}` : ''}`,
        retriedCount 
      };
    } catch (error) {
      this.logger.error('Failed to retry failed jobs:', error);
      throw error;
    }
  }

  async getQueueMetrics() {
    try {
      const stats = await this.getQueueStats();
      const queueStats = stats[QUEUE_NAMES.MEDICAL_PROCESSING];
      
      // Get recent completed jobs for timing analysis
      const recentJobs = await this.medicalQueue.getJobs(['completed'], 0, 99);
      const processingTimes = recentJobs
        .filter(job => job.finishedOn && job.processedOn)
        .map(job => job.finishedOn! - job.processedOn!);
      
      const averageProcessingTime = processingTimes.length > 0
        ? processingTimes.reduce((sum, time) => sum + time, 0) / processingTimes.length
        : 0;

      return {
        queueName: QUEUE_NAMES.MEDICAL_PROCESSING,
        totalJobs: Object.values(queueStats.total).reduce((sum: number, count: number) => sum + count, 0),
        averageProcessingTime,
        jobTypeBreakdown: queueStats.byJobType,
        currentCounts: queueStats.total,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      this.logger.error('Failed to get queue metrics:', error);
      throw error;
    }
  }
}

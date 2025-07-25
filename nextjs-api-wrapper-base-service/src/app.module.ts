import { Module } from "@nestjs/common";
import { ConfigModule } from "@nestjs/config";
import { BullModule } from "@nestjs/bull";
import { ThrottlerModule } from "@nestjs/throttler";
import { DocumentModule } from "./modules/document/document.module";
import { ProcessingModule } from "./modules/processing/processing.module";
import { HealthModule } from "./modules/health/health.module";
import { RedisConfigService } from "./config/redis.config";

@Module({
  imports: [
    // Configuration
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: ".env",
    }),

    // Throttling
    ThrottlerModule.forRootAsync({
      useFactory: () => ({
        throttlers: [
          {
            ttl: parseInt(process.env.THROTTLE_TTL) || 60,
            limit: parseInt(process.env.THROTTLE_LIMIT) || 100,
          },
        ],
      }),
    }),

    // BullMQ for job queues
    BullModule.forRootAsync({
      useClass: RedisConfigService,
    }),

    // Feature modules
    DocumentModule,
    ProcessingModule,
    HealthModule,
  ],
  providers: [RedisConfigService],
})
export class AppModule {}

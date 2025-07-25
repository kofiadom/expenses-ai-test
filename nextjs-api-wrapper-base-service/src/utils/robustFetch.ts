import { z, ZodError } from "zod";

export type RobustFetchParams<Schema extends z.Schema<any>> = {
  url: string;
  method?: "GET" | "POST" | "DELETE" | "PUT";
  body?: any;
  headers?: Record<string, string>;
  schema?: Schema;
  tryCount?: number;
  tryCooldown?: number;
};

export async function robustFetch<
  Schema extends z.Schema<any>,
  Output = z.infer<Schema>,
>({
  url,
  method = "GET",
  body,
  headers,
  schema,
  tryCount = 3,
  tryCooldown = 1000,
}: RobustFetchParams<Schema>): Promise<Output> {
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= tryCount; attempt++) {
    try {
      const response = await fetch(url, {
        method,
        headers: {
          ...(body instanceof FormData
            ? {}
            : body !== undefined
              ? { "Content-Type": "application/json" }
              : {}),
          ...(headers || {}),
        },
        ...(body instanceof FormData
          ? { body }
          : body !== undefined
            ? { body: JSON.stringify(body) }
            : {}),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const responseText = await response.text();
      let data: Output;

      try {
        data = JSON.parse(responseText);
      } catch (error) {
        throw new Error("Invalid JSON response");
      }

      if (schema) {
        try {
          data = schema.parse(data);
        } catch (error) {
          if (error instanceof ZodError) {
            throw new Error(`Schema validation failed: ${error.message}`);
          }
          throw error;
        }
      }

      return data;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      
      if (attempt < tryCount) {
        console.debug(`Request failed (attempt ${attempt}/${tryCount}), retrying in ${tryCooldown}ms...`);
        await new Promise((resolve) => setTimeout(resolve, tryCooldown));
      }
    }
  }

  throw lastError || new Error("Request failed after all retries");
}

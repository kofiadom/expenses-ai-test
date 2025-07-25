/**
 * Simple Queue implementation to replace yocto-queue dependency
 */
class SimpleQueue<T> {
  private items: T[] = [];

  enqueue(item: T): void {
    this.items.push(item);
  }

  dequeue(): T | undefined {
    return this.items.shift();
  }

  get size(): number {
    return this.items.length;
  }

  clear(): void {
    this.items = [];
  }
}

/**
 * Custom p-limit implementation compatible with our environment
 */
export default function pLimit(concurrency: number) {
  validateConcurrency(concurrency);

  const queue = new SimpleQueue<() => void>();
  let activeCount = 0;

  const resumeNext = () => {
    if (activeCount < concurrency && queue.size > 0) {
      const next = queue.dequeue();
      if (next) {
        next();
        // Since `pendingCount` has been decreased by one, increase `activeCount` by one.
        activeCount++;
      }
    }
  };

  const next = () => {
    activeCount--;
    resumeNext();
  };

  const run = async (function_: (...args: any[]) => any, resolve: (value: any) => void, arguments_: any[]) => {
    const result = (async () => function_(...arguments_))();

    resolve(result);

    try {
      await result;
    } catch {}

    next();
  };

  const enqueue = (function_: (...args: any[]) => any, resolve: (value: any) => void, arguments_: any[]) => {
    // Queue `internalResolve` instead of the `run` function
    // to preserve asynchronous context.
    new Promise<void>(internalResolve => {
      queue.enqueue(internalResolve);
    }).then(
      () => run(function_, resolve, arguments_),
    );

    (async () => {
      // This function needs to wait until the next microtask before comparing
      // `activeCount` to `concurrency`, because `activeCount` is updated asynchronously
      // after the `internalResolve` function is dequeued and called. The comparison in the if-statement
      // needs to happen asynchronously as well to get an up-to-date value for `activeCount`.
      await Promise.resolve();

      if (activeCount < concurrency) {
        resumeNext();
      }
    })();
  };

  const generator = (function_: (...args: any[]) => any, ...arguments_: any[]) => new Promise(resolve => {
    enqueue(function_, resolve, arguments_);
  });

  Object.defineProperties(generator, {
    activeCount: {
      get: () => activeCount,
    },
    pendingCount: {
      get: () => queue.size,
    },
    clearQueue: {
      value() {
        queue.clear();
      },
    },
    concurrency: {
      get: () => concurrency,

      set(newConcurrency: number) {
        validateConcurrency(newConcurrency);
        concurrency = newConcurrency;

        queueMicrotask(() => {
          // eslint-disable-next-line no-unmodified-loop-condition
          while (activeCount < concurrency && queue.size > 0) {
            resumeNext();
          }
        });
      },
    },
  });

  return generator;
}

export function limitFunction(function_: (...args: any[]) => any, option: { concurrency: number }) {
  const { concurrency } = option;
  const limit = pLimit(concurrency);

  return (...arguments_: any[]) => limit(() => function_(...arguments_));
}

function validateConcurrency(concurrency: number) {
  if (!((Number.isInteger(concurrency) || concurrency === Number.POSITIVE_INFINITY) && concurrency > 0)) {
    throw new TypeError('Expected `concurrency` to be a number from 1 and up');
  }
}

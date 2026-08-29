/**
 * DSH fork 谱系存储。
 *
 * 子会话只记录自己新增的 L0；读路径通过 ancestors() 获得“当前 + 父链”，
 * 从而继承父历史但永不看见兄弟分支。失败时退化为当前会话单点可见。
 */
import * as path from 'node:path';
import type { Session } from '@deepseek-ai/dsh-session';
import type { MemoryLogger, SessionLineage } from '../types.js';
import { atomicWriteJson, ensureDir, readJsonIfExists } from './io.js';

interface LineageFile {
  version: 1;
  sessions: Record<string, SessionLineage>;
}

const MAX_ENTRIES = 10_000;

export class SessionLineageStore {
  private readonly file: string;
  private readonly entries = new Map<string, SessionLineage>();
  private writeChain: Promise<void> = Promise.resolve();
  private persistFailed = false;

  constructor(dataDir: string, private readonly logger?: MemoryLogger) {
    this.file = path.join(dataDir, 'session-lineage.json');
  }

  async init(): Promise<void> {
    const data = await readJsonIfExists<Partial<LineageFile>>(this.file);
    if (!data?.sessions || typeof data.sessions !== 'object') return;
    for (const [sid, raw] of Object.entries(data.sessions)) {
      if (!raw || typeof raw !== 'object') continue;
      const parent = typeof raw.parentSessionId === 'string' ? raw.parentSessionId : undefined;
      this.entries.set(sid, {
        sessionId: sid,
        rootSessionId: typeof raw.rootSessionId === 'string' ? raw.rootSessionId : sid,
        parentSessionId: parent,
        seedLength: Number.isSafeInteger(raw.seedLength) ? raw.seedLength : undefined,
        createdAt: Number.isFinite(raw.createdAt) ? raw.createdAt : Date.now(),
      });
    }
  }

  /** 观察 live session；重复调用幂等，并会在稍后发现父链时修正 root。 */
  observe(session: Session): SessionLineage {
    const sid = String(session.id);
    const parent = session.header.parentSession ? String(session.header.parentSession) : undefined;
    const root = parent ? (this.entries.get(parent)?.rootSessionId ?? parent) : sid;
    const next: SessionLineage = {
      sessionId: sid,
      rootSessionId: root,
      parentSessionId: parent,
      seedLength: session.header.seedLength,
      createdAt: session.header.createdAt,
    };
    const old = this.entries.get(sid);
    if (!old || JSON.stringify(old) !== JSON.stringify(next)) {
      this.entries.set(sid, next);
      this.repairDescendantRoots(sid);
      this.queuePersist();
    }
    return this.entries.get(sid) ?? next;
  }

  get(sessionId: string): SessionLineage {
    return this.entries.get(sessionId) ?? {
      sessionId,
      rootSessionId: sessionId,
      createdAt: Date.now(),
    };
  }

  /** 当前会话到根会话，遇损坏环路安全截断。 */
  ancestors(sessionId: string): string[] {
    const out: string[] = [];
    const seen = new Set<string>();
    let cursor: string | undefined = sessionId;
    while (cursor && !seen.has(cursor)) {
      seen.add(cursor);
      out.push(cursor);
      cursor = this.entries.get(cursor)?.parentSessionId;
    }
    return out;
  }

  isFork(sessionId: string): boolean {
    return !!this.entries.get(sessionId)?.parentSessionId;
  }

  flush(): Promise<void> {
    return this.writeChain;
  }

  private repairDescendantRoots(parentId: string): void {
    const root = this.entries.get(parentId)?.rootSessionId ?? parentId;
    const queue = [parentId];
    const seen = new Set(queue);
    while (queue.length > 0) {
      const p = queue.shift()!;
      for (const [sid, entry] of this.entries) {
        if (entry.parentSessionId !== p || seen.has(sid)) continue;
        seen.add(sid);
        entry.rootSessionId = root;
        queue.push(sid);
      }
    }
  }

  private queuePersist(): void {
    this.writeChain = this.writeChain.then(() => this.persist());
  }

  private async persist(): Promise<void> {
    try {
      await ensureDir(path.dirname(this.file));
      while (this.entries.size > MAX_ENTRIES) {
        let oldest: string | undefined;
        let oldestAt = Infinity;
        for (const [sid, e] of this.entries) {
          if (e.createdAt < oldestAt) { oldest = sid; oldestAt = e.createdAt; }
        }
        if (!oldest) break;
        this.entries.delete(oldest);
      }
      const sessions: Record<string, SessionLineage> = {};
      for (const [sid, entry] of this.entries) sessions[sid] = entry;
      await atomicWriteJson(this.file, { version: 1, sessions } satisfies LineageFile);
      this.persistFailed = false;
    } catch (err) {
      if (!this.persistFailed) {
        this.persistFailed = true;
        this.logger?.warn(`[memory] 会话谱系持久化失败（降级内存态）: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }
}

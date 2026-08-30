/**
 * 压缩归档：把离开活跃上下文的会话按 12 小时桶切成语义索引。
 * 梗概只负责“想起该去哪里找”，原文证据始终留在 L0。
 */
import type { Context } from '@deepseek-ai/cordis';
import { createHash } from 'node:crypto';
import { promises as fs } from 'node:fs';
import * as path from 'node:path';
import type { MemoryConfig } from '../config.js';
import { callLLM } from '../llm.js';
import type { ArchiveSearchHit, ArchiveSegmentRecord, ConversationMessage, MemoryLogger } from '../types.js';
import { sanitizeText } from '../util/sanitize.js';
import { tokenizeForSearch } from './search-utils.js';
import type { L0Store } from './l0.js';
import { appendJsonl, dayKey, ensureDir, readJsonl } from './io.js';
import type { MemoryDb } from './sqlite.js';

export const ARCHIVE_PERIOD_MS = 12 * 60 * 60 * 1000;
export const ARCHIVE_SUMMARY_VERSION = 1;
const ARCHIVE_SEARCH_CANDIDATES = 200;

/** 摘要异常超长时在完整行/句边界收口，避免留下半个 Markdown 条目。 */
export function truncateArchiveSummary(text: string, maxChars: number): string {
  const clean = text.trim();
  if (clean.length <= maxChars) return clean;
  const head = clean.slice(0, Math.max(1, maxChars - 1));
  const floor = Math.floor(maxChars * 0.5);
  const line = head.lastIndexOf('\n');
  const sentence = Math.max(head.lastIndexOf('。'), head.lastIndexOf('！'), head.lastIndexOf('？'), head.lastIndexOf(';'));
  const boundary = Math.max(line >= floor ? line : -1, sentence >= floor ? sentence + 1 : -1);
  return `${head.slice(0, boundary > 0 ? boundary : head.length).trimEnd()}…`;
}

function archiveId(sessionId: string, messages: readonly ConversationMessage[]): string {
  const first = messages[0]?.id ?? '';
  const last = messages[messages.length - 1]?.id ?? '';
  return `archive_${createHash('sha256').update(`${sessionId}\0${first}\0${last}`).digest('hex').slice(0, 24)}`;
}

function sourceText(messages: readonly ConversationMessage[]): string {
  return messages.map((message) => {
    const role = message.role === 'user' ? '用户' : '助手';
    return `[${new Date(message.timestamp).toISOString()}] ${role}: ${sanitizeText(message.content)}`;
  }).join('\n');
}

/** 只在消息边界切分，避免摘要输入截断半条原话。 */
export function packArchiveSegments(
  sessionId: string,
  messages: readonly ConversationMessage[],
  maxChars: number,
  now = Date.now(),
): ArchiveSegmentRecord[] {
  const byBucket = new Map<number, ConversationMessage[]>();
  for (const message of [...messages].sort((a, b) => a.timestamp - b.timestamp)) {
    const bucket = Math.floor(message.timestamp / ARCHIVE_PERIOD_MS) * ARCHIVE_PERIOD_MS;
    const list = byBucket.get(bucket) ?? [];
    list.push(message);
    byBucket.set(bucket, list);
  }

  const out: ArchiveSegmentRecord[] = [];
  for (const [bucketStart, bucketMessages] of [...byBucket.entries()].sort((a, b) => a[0] - b[0])) {
    let current: ConversationMessage[] = [];
    let currentChars = 0;
    const flush = (): void => {
      if (current.length === 0) return;
      const text = sourceText(current);
      out.push({
        id: archiveId(sessionId, current),
        sessionId,
        bucketStart,
        latestAt: current[current.length - 1].timestamp,
        summary: '',
        sourceText: text.slice(0, maxChars),
        messageIds: current.map((message) => message.id),
        status: 'pending',
        summaryVersion: ARCHIVE_SUMMARY_VERSION,
        createdAt: now,
        updatedAt: now,
      });
      current = [];
      currentChars = 0;
    };
    for (const message of bucketMessages) {
      const nextChars = message.content.length + 64;
      if (current.length > 0 && currentChars + nextChars > maxChars) flush();
      current.push(message);
      currentChars += nextChars;
    }
    flush();
  }
  return out;
}

export class ArchiveStore {
  private readonly dir: string;
  private queue: Promise<void> = Promise.resolve();
  private readonly queued = new Set<string>();
  private disposed = false;

  constructor(
    dataDir: string,
    private readonly db: MemoryDb,
    private readonly l0: L0Store,
    private readonly ctx: Context,
    private readonly cfg: () => MemoryConfig,
    private readonly logger: MemoryLogger,
  ) {
    this.dir = path.join(dataDir, 'archives');
  }

  async init(): Promise<void> {
    await ensureDir(this.dir);
    // JSONL 是追加事实源；按文件/行顺序回灌，后写版本覆盖 pending 版本。
    const files = (await fs.readdir(this.dir).catch(() => [] as string[])).filter((name) => name.endsWith('.jsonl')).sort();
    for (const name of files) {
      const records = await readJsonl<ArchiveSegmentRecord>(path.join(this.dir, name));
      for (const record of records) {
        if (record?.id && record.sessionId && Array.isArray(record.messageIds)) this.db.upsertArchive(record);
      }
    }
    for (const record of this.db.listArchivesNeedingSummary(ARCHIVE_SUMMARY_VERSION)) this.enqueue(record);
  }

  /**
   * 压缩路径只等待本地持久化：L0 回填与 pending 梗概均先落盘，LLM 摘要在后台串行执行。
   */
  async ingest(sessionId: string, messages: ConversationMessage[]): Promise<{ backfilled: number; segments: number }> {
    if (this.disposed || messages.length === 0) return { backfilled: 0, segments: 0 };
    const cfg = this.cfg();
    const backfilled = await this.l0.appendMissing(sessionId, messages, false);
    let segments = 0;
    for (const candidate of packArchiveSegments(sessionId, messages, cfg.archive.maxSegmentChars)) {
      const existing = this.db.getArchive(candidate.id);
      if (existing?.status === 'ready') continue;
      const record = existing ?? candidate;
      if (!this.db.upsertArchive(record)) continue;
      await appendJsonl(path.join(this.dir, `${dayKey(record.bucketStart)}.jsonl`), [record]);
      this.enqueue(record);
      segments++;
    }
    if (backfilled > 0 || segments > 0) {
      this.logger.info(`[memory] 压缩归档落盘 session=${sessionId}：L0 回填 ${backfilled} 条，待摘要 ${segments} 段`);
    }
    return { backfilled, segments };
  }

  search(
    query: string,
    limit: number,
    visibleSessionIds?: readonly string[],
    strict = false,
  ): ArchiveSearchHit[] {
    if (limit <= 0) return [];
    const visible = visibleSessionIds ? new Set(visibleSessionIds) : undefined;
    const queryTokens = tokenizeForSearch(query);
    return this.db.searchArchiveFts(query, ARCHIVE_SEARCH_CANDIDATES)
      .filter((hit) => !visible || visible.has(hit.sessionId))
      .filter((hit) => {
        if (!strict || queryTokens.length === 0) return true;
        const hitTokens = new Set(tokenizeForSearch(hit.summary));
        const matched = queryTokens.reduce((n, token) => n + (hitTokens.has(token) ? 1 : 0), 0);
        return matched >= Math.max(1, Math.ceil(queryTokens.length * 0.2));
      })
      .slice(0, limit);
  }

  get(id: string): ArchiveSegmentRecord | undefined {
    return this.db.getArchive(id);
  }

  messages(id: string) {
    const record = this.db.getArchive(id);
    return record ? this.l0.getByIds(record.messageIds) : [];
  }

  referencedMessageIds(): Set<string> {
    return this.db.archiveReferencedMessageIds();
  }

  flush(): Promise<void> {
    return this.queue;
  }

  close(): void {
    this.disposed = true;
  }

  private enqueue(record: ArchiveSegmentRecord): void {
    if (this.disposed || this.queued.has(record.id)) return;
    this.queued.add(record.id);
    this.queue = this.queue
      .then(() => this.summarize(record))
      .catch((err) => this.logger.warn(`[memory] 压缩归档摘要失败 id=${record.id}（保留 pending 下次重试）: ${err instanceof Error ? err.message : String(err)}`))
      .finally(() => this.queued.delete(record.id));
  }

  private async summarize(record: ArchiveSegmentRecord): Promise<void> {
    if (this.disposed) return;
    const cfg = this.cfg();
    const raw = await callLLM(this.ctx, cfg, {
      system: [
        '你是会话档案索引器。把一段旧对话压缩成可供语义检索的中文梗概。',
        '目标是帮助未来模型判断“这里可能有答案”，而不是替代原文。',
        '只写已出现的事实、事件、决定、称呼、情绪脉络、未完话题和有辨识度的关键词。',
        `正文控制在 ${Math.max(200, cfg.archive.maxSummaryChars - 100)} 字以内，优先覆盖不同主题，不要在单一主题上展开。`,
        '不要记录模型内部推理、工具噪声或任何凭据的值、过期时间、泄露细节；若安全事件本身重要，只写“曾处理敏感凭据，细节需回查原文”。',
        '不要补造事实。',
        '输出简洁 Markdown 要点，不要解释任务。',
      ].join('\n'),
      user: `时间范围：${new Date(record.bucketStart).toISOString()} 至 ${new Date(record.latestAt).toISOString()}\n\n${record.sourceText}`,
      maxTokens: 2048,
      temperature: 0.1,
      logger: this.logger,
    });
    if (this.disposed) return;
    const summary = truncateArchiveSummary(raw, cfg.archive.maxSummaryChars);
    if (!summary) throw new Error('摘要为空');
    const ready: ArchiveSegmentRecord = {
      ...record,
      summary,
      status: 'ready',
      summaryVersion: ARCHIVE_SUMMARY_VERSION,
      updatedAt: Date.now(),
    };
    if (!this.db.upsertArchive(ready)) throw new Error('SQLite 写入失败');
    await appendJsonl(path.join(this.dir, `${dayKey(ready.bucketStart)}.jsonl`), [ready]);
    this.logger.info(`[memory] 压缩归档摘要完成 id=${ready.id}（${ready.messageIds.length} 条消息 → ${summary.length} 字）`);
  }
}

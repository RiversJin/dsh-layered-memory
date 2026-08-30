/**
 * 压缩归档：把离开活跃上下文的会话按 12 小时桶切成语义索引。
 * 梗概只负责“想起该去哪里找”，原文证据始终留在 L0。
 */
import type { Context } from '@deepseek-ai/cordis';
import type { MemoryConfig } from '../config.js';
import type { ArchiveSearchHit, ArchiveSegmentRecord, ConversationMessage, MemoryLogger } from '../types.js';
import type { L0Store } from './l0.js';
import type { MemoryDb } from './sqlite.js';
export declare const ARCHIVE_PERIOD_MS: number;
export declare const ARCHIVE_SUMMARY_VERSION = 1;
/** 摘要异常超长时在完整行/句边界收口，避免留下半个 Markdown 条目。 */
export declare function truncateArchiveSummary(text: string, maxChars: number): string;
/** 只在消息边界切分，避免摘要输入截断半条原话。 */
export declare function packArchiveSegments(sessionId: string, messages: readonly ConversationMessage[], maxChars: number, now?: number): ArchiveSegmentRecord[];
export declare class ArchiveStore {
    private readonly db;
    private readonly l0;
    private readonly ctx;
    private readonly cfg;
    private readonly logger;
    private readonly dir;
    private queue;
    private readonly queued;
    private disposed;
    constructor(dataDir: string, db: MemoryDb, l0: L0Store, ctx: Context, cfg: () => MemoryConfig, logger: MemoryLogger);
    init(): Promise<void>;
    /**
     * 压缩路径只等待本地持久化：L0 回填与 pending 梗概均先落盘，LLM 摘要在后台串行执行。
     */
    ingest(sessionId: string, messages: ConversationMessage[]): Promise<{
        backfilled: number;
        segments: number;
    }>;
    search(query: string, limit: number, visibleSessionIds?: readonly string[], strict?: boolean): ArchiveSearchHit[];
    get(id: string): ArchiveSegmentRecord | undefined;
    messages(id: string): import("../types.js").L0MessageRecord[];
    referencedMessageIds(): Set<string>;
    flush(): Promise<void>;
    close(): void;
    private enqueue;
    private summarize;
}

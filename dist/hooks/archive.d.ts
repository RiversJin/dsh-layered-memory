/** 压缩事件 → L0 回填 + 独立语义档案。 */
import type { Context } from '@deepseek-ai/cordis';
import type { Session } from '@deepseek-ai/dsh-session';
import type { MemoryConfig } from '../config.js';
import type { ArchiveStore } from '../store/archive.js';
import type { SessionLineageStore } from '../store/session-lineage.js';
import type { SessionModeStore } from '../store/session-modes.js';
import type { ConversationMessage, MemoryLogger } from '../types.js';
interface OwnedMessage {
    owner: string;
    message: ConversationMessage;
}
/**
 * fork seed 的时间早于子会话 createdAt：归到父分支；子会话新消息归自己。
 * 根会话即使由历史导入而 createdAt 较新，也仍全部归根会话，不误分支。
 */
export declare function archiveMessagesFromSession(session: Session, compactionAt: number, cfg: MemoryConfig, lineage: SessionLineageStore): OwnedMessage[];
export declare function registerArchiveCapture(ctx: Context, cfg: MemoryConfig, archive: ArchiveStore, logger: MemoryLogger, modes: SessionModeStore, lineage: SessionLineageStore): (() => Promise<void>) | undefined;
export {};

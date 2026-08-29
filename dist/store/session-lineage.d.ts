import type { Session } from '@deepseek-ai/dsh-session';
import type { MemoryLogger, SessionLineage } from '../types.js';
export declare class SessionLineageStore {
    private readonly logger?;
    private readonly file;
    private readonly entries;
    private writeChain;
    private persistFailed;
    constructor(dataDir: string, logger?: MemoryLogger | undefined);
    init(): Promise<void>;
    /** 观察 live session；重复调用幂等，并会在稍后发现父链时修正 root。 */
    observe(session: Session): SessionLineage;
    get(sessionId: string): SessionLineage;
    /** 当前会话到根会话，遇损坏环路安全截断。 */
    ancestors(sessionId: string): string[];
    isFork(sessionId: string): boolean;
    /** 当前会话的 agent preset；旧 lineage 会在 live session 被 observe 后补齐。 */
    presetOf(sessionId: string): string | undefined;
    flush(): Promise<void>;
    private repairDescendantRoots;
    private queuePersist;
    private persist;
}

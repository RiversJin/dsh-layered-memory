/**
 * DSH fork 谱系存储。
 *
 * 子会话只记录自己新增的 L0；读路径通过 ancestors() 获得“当前 + 父链”，
 * 从而继承父历史但永不看见兄弟分支。失败时退化为当前会话单点可见。
 */
import * as path from 'node:path';
import { atomicWriteJson, ensureDir, readJsonIfExists } from './io.js';
const MAX_ENTRIES = 10_000;
export class SessionLineageStore {
    logger;
    file;
    entries = new Map();
    writeChain = Promise.resolve();
    persistFailed = false;
    constructor(dataDir, logger) {
        this.logger = logger;
        this.file = path.join(dataDir, 'session-lineage.json');
    }
    async init() {
        const data = await readJsonIfExists(this.file);
        if (!data?.sessions || typeof data.sessions !== 'object')
            return;
        for (const [sid, raw] of Object.entries(data.sessions)) {
            if (!raw || typeof raw !== 'object')
                continue;
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
    observe(session) {
        const sid = String(session.id);
        const parent = session.header.parentSession ? String(session.header.parentSession) : undefined;
        const root = parent ? (this.entries.get(parent)?.rootSessionId ?? parent) : sid;
        const next = {
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
    get(sessionId) {
        return this.entries.get(sessionId) ?? {
            sessionId,
            rootSessionId: sessionId,
            createdAt: Date.now(),
        };
    }
    /** 当前会话到根会话，遇损坏环路安全截断。 */
    ancestors(sessionId) {
        const out = [];
        const seen = new Set();
        let cursor = sessionId;
        while (cursor && !seen.has(cursor)) {
            seen.add(cursor);
            out.push(cursor);
            cursor = this.entries.get(cursor)?.parentSessionId;
        }
        return out;
    }
    isFork(sessionId) {
        return !!this.entries.get(sessionId)?.parentSessionId;
    }
    flush() {
        return this.writeChain;
    }
    repairDescendantRoots(parentId) {
        const root = this.entries.get(parentId)?.rootSessionId ?? parentId;
        const queue = [parentId];
        const seen = new Set(queue);
        while (queue.length > 0) {
            const p = queue.shift();
            for (const [sid, entry] of this.entries) {
                if (entry.parentSessionId !== p || seen.has(sid))
                    continue;
                seen.add(sid);
                entry.rootSessionId = root;
                queue.push(sid);
            }
        }
    }
    queuePersist() {
        this.writeChain = this.writeChain.then(() => this.persist());
    }
    async persist() {
        try {
            await ensureDir(path.dirname(this.file));
            while (this.entries.size > MAX_ENTRIES) {
                let oldest;
                let oldestAt = Infinity;
                for (const [sid, e] of this.entries) {
                    if (e.createdAt < oldestAt) {
                        oldest = sid;
                        oldestAt = e.createdAt;
                    }
                }
                if (!oldest)
                    break;
                this.entries.delete(oldest);
            }
            const sessions = {};
            for (const [sid, entry] of this.entries)
                sessions[sid] = entry;
            await atomicWriteJson(this.file, { version: 1, sessions });
            this.persistFailed = false;
        }
        catch (err) {
            if (!this.persistFailed) {
                this.persistFailed = true;
                this.logger?.warn(`[memory] 会话谱系持久化失败（降级内存态）: ${err instanceof Error ? err.message : String(err)}`);
            }
        }
    }
}

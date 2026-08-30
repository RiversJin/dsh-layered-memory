import { ARCHIVE_PERIOD_MS } from '../store/archive.js';
import { sanitizeText, shouldCaptureL0, stripCodeBlocks } from '../util/sanitize.js';
import { blocksToText } from '../util/text.js';
/**
 * fork seed 的时间早于子会话 createdAt：归到父分支；子会话新消息归自己。
 * 根会话即使由历史导入而 createdAt 较新，也仍全部归根会话，不误分支。
 */
export function archiveMessagesFromSession(session, compactionAt, cfg, lineage) {
    const entry = lineage.observe(session);
    const sid = String(session.id);
    const closedBefore = Math.floor(compactionAt / ARCHIVE_PERIOD_MS) * ARCHIVE_PERIOD_MS;
    const out = [];
    for (const event of session.events) {
        if (event.time >= closedBefore)
            continue;
        const owner = entry.parentSessionId && event.time < entry.createdAt ? entry.parentSessionId : sid;
        let role;
        let content = '';
        if (event.type === 'user/message') {
            const message = event.data;
            if (message.source?.kind !== 'user')
                continue;
            role = 'user';
            content = sanitizeText(blocksToText(message.content, false));
        }
        else if (event.type === 'assistant/message') {
            const data = event.data;
            role = 'assistant';
            content = sanitizeText(blocksToText(data.message?.content, false));
            if (cfg.capture.stripCodeBlocks)
                content = stripCodeBlocks(content);
        }
        else {
            continue;
        }
        if (!shouldCaptureL0(content))
            continue;
        out.push({
            owner,
            message: {
                id: `l0:${owner}:${event.seq}`,
                role,
                content: content.slice(0, cfg.capture.maxMessageChars),
                timestamp: event.time,
            },
        });
    }
    return out;
}
export function registerArchiveCapture(ctx, cfg, archive, logger, modes, lineage) {
    if (!cfg.archive.enabled)
        return;
    let queue = Promise.resolve();
    ctx.on('session/event', (session, event) => {
        // dsh-session 的公开 SessionEvent 联合暂未列出 compaction/*，但运行时通过同一
        // session/event 总线发布；这里仅扩宽判别字段，不假定 data 形状。
        const observed = event;
        if (observed.type !== 'compaction/summary')
            return;
        const sid = String(session.id);
        if (modes.get(sid) === 'off')
            return;
        const owned = archiveMessagesFromSession(session, observed.time, cfg, lineage);
        const byOwner = new Map();
        for (const { owner, message } of owned) {
            const list = byOwner.get(owner) ?? [];
            list.push(message);
            byOwner.set(owner, list);
        }
        queue = queue.then(async () => {
            for (const [owner, messages] of byOwner)
                await archive.ingest(owner, messages);
        }).catch((err) => {
            logger.warn(`[memory] 压缩归档事件处理失败（不影响压缩）: ${err instanceof Error ? err.message : String(err)}`);
        });
    });
    return async () => {
        await queue;
        await archive.flush();
    };
}

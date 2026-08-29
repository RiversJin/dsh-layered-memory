/**
 * 模型可调用工具：memory_commit（显式写 L1）、memory_search（L1）、
 * conversation_search（L0）、memory_read_scene（L2/L3）。
 *
 * 会话档位联动：execute 的 exec.agent 即发起调用的 agent（agent.id === sessionId），
 * memory_search 按会话档位过滤族（auto 不过滤，纯档只查本族）；off 档下三工具统一
 * 返回提示（本会话已对记忆系统隐身）。conversation_search 检索范围保持全库。
 * 只写会话（#38：注入覆盖=关）同款拒读，notice 区分文案——写入走捕获钩子不经工具，
 * 拒读不影响"只写"语义。
 */
import type { Context } from '@deepseek-ai/cordis';
import { randomBytes } from 'node:crypto';
import { defineTool } from '@deepseek-ai/dsh-tools';
import type { MemoryConfig } from '../config.js';
import type { LiveSettingsHandle } from '../settings.js';
import type { L0Store } from '../store/l0.js';
import type { L1Store } from '../store/l1.js';
import type { PersonaStore } from '../store/persona.js';
import type { SceneStore } from '../store/scenes.js';
import type { SessionModeStore } from '../store/session-modes.js';
import type { SessionLineageStore } from '../store/session-lineage.js';
import type { MemoryFamily, MemoryLogger, MemoryRecord, MemoryScope } from '../types.js';
import { familyForType } from '../types.js';

const OFF_NOTICE = '本会话的记忆档位为"关闭"：该会话对记忆系统完全隐身，不读取也不写入记忆。';
const WRITE_ONLY_NOTICE = '本会话为只写模式：记忆照常沉淀，但不读取。';
const GLOBAL_OFF_NOTICE = '记忆注入已全局停用：本会话不读取记忆（沉淀照常）。';

export function registerMemoryTools(
  ctx: Context,
  cfg: MemoryConfig,
  stores: {
    l0: L0Store;
    l1: L1Store;
    scenes: Record<MemoryFamily, SceneStore>;
    persona: Record<MemoryFamily, PersonaStore>;
  },
  logger: MemoryLogger,
  modes: SessionModeStore,
  live: LiveSettingsHandle,
  lineage?: SessionLineageStore,
): void {
  if (!cfg.tools) return;

  /**
   * 调用会话的检索族（auto → undefined 不过滤；off/只写 → null 表示整体禁用）。
   * fail-open：exec.agent 缺失（宿主调用路径未带 agent 标识）按全族检索放行——
   * 档位隔离依赖宿主正确传递 exec.agent.id，缺失只告警一次不拒绝工具调用。
   */
  let warnedNoAgent = false;
  const familyOfCaller = (agentId: string | undefined): MemoryFamily | undefined | null => {
    if (agentId === undefined) {
      if (!warnedNoAgent) {
        warnedNoAgent = true;
        logger.warn('[memory] 工具调用缺少 agent 标识（exec.agent 未传递），档位过滤退化为全族检索');
      }
      return undefined;
    }
    const mode = modes.get(agentId);
    if (mode === 'off') return null;
    // 只写会话拒读（#38，T2 裁决）：与注入同属读维度，不拒则"不注入"从工具路径漏风
    if (!modes.resolvedRecall(agentId, live.get().recall)) return null;
    return mode === 'auto' ? undefined : mode;
  };

  const visibleSessions = (agentId: string | undefined): string[] | undefined =>
    agentId === undefined ? undefined : (lineage?.ancestors(agentId) ?? [agentId]);
  const presetOf = (agentId: string | undefined): string | undefined =>
    agentId === undefined ? undefined : lineage?.presetOf(agentId);
  const presetBinding = (presetId: string): string => `preset:${presetId}`;

  // ── memory_commit: 模型显式写入一条已整理的 L1 原子记忆 ──
  ctx.tools.register(
    defineTool({
      name: 'memory_commit',
      description:
        '显式保存一条值得跨轮次保留的原子记忆。仅在信息稳定、未来有用且用户允许保留时调用；不要保存临时闲聊、工具输出、秘密或模型推理。若要修订旧记忆，先 memory_search 获取 id，再传 replaces。',
      parameters: {
        content: { type: 'string', required: true, description: '自包含、可独立理解的一条事实或规则' },
        type: { type: 'string', required: true, description: 'persona/episodic/instruction/work_fact/work_task/work_method/work_artifact' },
        priority: { type: 'number', description: '0-100，默认 60' },
        scope: { type: 'string', description: 'auto（默认）/global/preset/branch；global 对所有会话可见，preset 仅同 agent preset，branch 仅当前分支及其后代' },
        replaces: { type: 'string', description: '要被本条修订替换的旧 memory id；须先搜索确认' },
      },
      output: {
        schema: {
          type: 'object',
          properties: {
            status: { type: 'string' },
            id: { type: 'string' },
            scope: { type: 'string' },
            notice: { type: 'string' },
          },
          additionalProperties: false,
        },
        render: (_args, value) => [{ type: 'text', text: value.notice ?? `记忆已保存（${value.id ?? ''}，${value.scope ?? ''}）` }],
      },
      execute: async (args, exec) => {
        const sid = exec.agent?.id;
        if (!sid) return { status: 'rejected', id: '', scope: '', notice: '缺少会话标识，未写入记忆。' };
        if (modes.get(sid) === 'off') return { status: 'rejected', id: '', scope: '', notice: OFF_NOTICE };
        const content = args.content.trim();
        if (!content) return { status: 'rejected', id: '', scope: '', notice: '记忆内容为空，未写入。' };
        const type = args.type.trim() || 'episodic';
        const family = familyForType(type);
        const rawScope = args.scope?.trim();
        const callerPreset = presetOf(sid);
        const scope: MemoryScope = rawScope === 'global' || rawScope === 'preset' || rawScope === 'branch'
          ? rawScope
          : ((type === 'persona' || type === 'instruction') && callerPreset ? 'preset'
            : (type === 'persona' || type === 'instruction') ? 'global' : 'branch');
        if (scope === 'preset' && !callerPreset) {
          return { status: 'rejected', id: '', scope, notice: '当前会话没有 agent preset，无法写入 preset 记忆。' };
        }
        const bindingSessionId = scope === 'preset' ? presetBinding(callerPreset!) : sid;
        const visible = lineage?.ancestors(sid) ?? [sid];

        // 精确语义重复幂等（忽略空白与大小写）；相似但不同的内容不擅自合并。
        const candidates = await stores.l1.searchCandidates(
          content,
          Math.max(stores.l1.size, 20),
          family,
          visible,
          callerPreset,
        );
        const norm = (s: string): string => s.replace(/\s+/g, ' ').trim().toLocaleLowerCase();
        const duplicate = candidates.find((r) =>
          r.type === type && (r.scope ?? 'global') === scope &&
          (scope !== 'preset' || r.sessionId === bindingSessionId) && norm(r.content) === norm(content));
        if (duplicate) {
          return { status: 'duplicate', id: duplicate.id, scope, notice: `已有相同记忆（${duplicate.id}），未重复写入。` };
        }

        let replaced: MemoryRecord | undefined;
        if (args.replaces?.trim()) {
          const target = stores.l1.getByIds([args.replaces.trim()])[0];
          const targetScope = target?.scope ?? 'global';
          const targetVisible = !!target && (
            targetScope === 'global' ||
            (targetScope === 'preset' && target.sessionId === presetBinding(callerPreset ?? '')) ||
            (targetScope === 'branch' && visible.includes(target.sessionId ?? 'default'))
          );
          if (!target || !targetVisible) {
            return { status: 'rejected', id: '', scope, notice: '指定的旧记忆不存在或当前分支不可见，未写入。' };
          }
          if ((target.family ?? familyForType(target.type)) !== family) {
            return { status: 'rejected', id: '', scope, notice: '新旧记忆分属不同记忆族，未执行替换。' };
          }
          if (targetScope !== scope) {
            return { status: 'rejected', id: '', scope, notice: '新旧记忆可见域不同，未执行替换。' };
          }
          if (targetScope === 'branch' && (target.sessionId ?? 'default') !== sid) {
            return { status: 'rejected', id: '', scope, notice: '祖先分支记忆在子分支中只读；请另存当前分支的新记忆。' };
          }
          replaced = target;
        }

        const now = Date.now();
        const record: MemoryRecord = {
          id: `mem_${now}_${randomBytes(3).toString('hex')}`,
          content,
          type,
          priority: Math.min(100, Math.max(0, Math.round(args.priority ?? 60))),
          scene_name: 'explicit-commit',
          timestamps: [now],
          createdAt: now,
          updatedAt: now,
          version: (replaced?.version ?? -1) + 1,
          source_message_ids: currentTurnSourceIds(exec.agent.session.events, sid),
          metadata: { committed_by: 'memory_commit' },
          sessionId: bindingSessionId,
          scope,
          family,
        };
        await stores.l1.appendNew([record]);
        if (replaced) await stores.l1.deleteBatch([replaced.id]);
        logger.info(`[memory] 显式提交 L1 id=${record.id} type=${type} scope=${scope} binding=${bindingSessionId} caller=${sid}${replaced ? ` replaces=${replaced.id}` : ''}`);
        return replaced
          ? { status: 'replaced', id: record.id, scope, notice: `记忆已修订（${record.id}）。` }
          : { status: 'stored', id: record.id, scope };
      },
    }),
  );

  /** 拒读时的归因文案（familyOfCaller 判 null 后重查内存 Map，成本可忽略）：
   *  off 完全隐身 / 会话只写覆盖 / 全局召回关——三种停用各说各话，不谎报只写。 */
  const blockNoticeOf = (agentId: string | undefined): string => {
    if (agentId !== undefined) {
      if (modes.get(agentId) === 'off') return OFF_NOTICE;
      if (modes.getRecall(agentId) === false) return WRITE_ONLY_NOTICE;
      if (!modes.resolvedRecall(agentId, live.get().recall)) return GLOBAL_OFF_NOTICE;
    }
    return OFF_NOTICE;
  };

  // ── memory_search: L1 结构化记忆 ──
  ctx.tools.register(
    defineTool({
      name: 'memory_search',
      description:
        '搜索结构化记忆（L1 原子记忆），适合用户偏好、历史事件、项目事实、任务、规则与工作方法；需要具体原话或时间线时改用 conversation_search。两种搜索每轮合计最多调用 3 次；仍无结果时直接按已有信息回答。',
      parameters: {
        query: { type: 'string', required: true, description: '搜索查询文本（自然语言）' },
        limit: { type: 'number', description: '最大返回条数（默认 5）' },
        type: { type: 'string', description: '按记忆类型过滤（如 persona/episodic/instruction/work_fact/work_task/work_method/work_artifact）' },
      },
      output: {
        schema: {
          type: 'object',
          properties: {
            items: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  content: { type: 'string' },
                  id: { type: 'string' },
                  type: { type: 'string' },
                  scene_name: { type: 'string' },
                  score: { type: 'number' },
                },
                additionalProperties: false,
              },
            },
            notice: { type: 'string', description: '非搜索结果的状态提示（如本会话记忆已关闭）' },
          },
          additionalProperties: false,
        },
        render: (_args, value) => [
          { type: 'text', text: value.notice ?? renderMemoryItems(value.items ?? []) },
        ],
      },
      execute: async (args, exec) => {
        const family = familyOfCaller(exec.agent?.id);
        if (family === null) return { items: [], notice: blockNoticeOf(exec.agent?.id) };
        const limit = Math.min(Math.max(args.limit ?? 5, 1), 20);
        const hits = await stores.l1.search(args.query, limit, {
          type: args.type || undefined,
          family: family ?? undefined,
          visibleSessionIds: visibleSessions(exec.agent?.id),
          visiblePresetId: presetOf(exec.agent?.id),
        });
        return {
          items: hits.map((h) => ({
            id: h.id,
            content: h.content,
            type: h.type,
            scene_name: h.scene_name,
            score: Math.round(h.score * 100) / 100,
          })),
        };
      },
    }),
  );

  // ── conversation_search: L0 原始对话 ──
  ctx.tools.register(
    defineTool({
      name: 'conversation_search',
      description:
        '搜索原始对话历史（L0），返回带时间戳的消息，适合具体原话、时间线与上下文细节；结构化偏好、事实、任务或规则优先用 memory_search。两种搜索每轮合计最多调用 3 次；仍无结果时直接按已有信息回答。',
      parameters: {
        query: { type: 'string', required: true, description: '搜索查询文本' },
        limit: { type: 'number', description: '最大返回条数（默认 5）' },
      },
      output: {
        schema: {
          type: 'object',
          properties: {
            items: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  session_id: { type: 'string' },
                  role: { type: 'string' },
                  content: { type: 'string' },
                  timestamp: { type: 'number' },
                },
                additionalProperties: false,
              },
            },
            notice: { type: 'string', description: '非搜索结果的状态提示（如本会话记忆已关闭）' },
          },
          additionalProperties: false,
        },
        render: (_args, value) => [
          { type: 'text', text: value.notice ?? renderConversationItems(value.items ?? []) },
        ],
      },
      execute: async (args, exec) => {
        if (familyOfCaller(exec.agent?.id) === null) return { items: [], notice: blockNoticeOf(exec.agent?.id) };
        const limit = Math.min(Math.max(args.limit ?? 5, 1), 20);
        const records = await stores.l0.search(args.query, limit, visibleSessions(exec.agent?.id));
        return {
          items: records.map((r) => ({
            session_id: r.sessionId,
            role: r.role,
            content: r.content,
            timestamp: r.timestamp,
          })),
        };
      },
    }),
  );

  // ── memory_read_scene: 读取 L2 场景块 / L3 画像 ──
  ctx.tools.register(
    defineTool({
      name: 'memory_read_scene',
      description:
        '读取记忆文件详情：L2 场景块（场景目录下的 .md 文件）或 L3 画像（persona-chat.md / persona-work.md）。返回文件完整内容。',
      parameters: {
        path: { type: 'string', required: true, description: '场景文件名，或 persona-chat.md / persona-work.md' },
      },
      output: {
        schema: {
          type: 'object',
          properties: {
            content: { type: 'string', description: '文件内容（不存在则为空字符串）' },
          },
          additionalProperties: false,
        },
        render: (_args, value) => [
          { type: 'text', text: value.content ? `\`\`\`markdown\n${value.content}\n\`\`\`` : '（文件不存在或为空）' },
        ],
      },
      execute: async (args, exec) => {
        if (familyOfCaller(exec.agent?.id) === null) return { content: blockNoticeOf(exec.agent?.id) };
        const p = args.path.trim();
        let content: string | undefined;
        if (p === 'persona.md' || p === 'persona-chat.md' || p === 'persona' || p === 'persona-chat') {
          content = await stores.persona.chat.read();
        } else if (p === 'persona-work.md' || p === 'persona-work') {
          content = await stores.persona.work.read();
        } else {
          // 场景文件在两族目录里按名查找（先本族后另一族）
          const primary = familyOfCaller(exec.agent?.id) ?? 'chat';
          const other: MemoryFamily = primary === 'chat' ? 'work' : 'chat';
          content =
            (await stores.scenes[primary].read(p)) ?? (await stores.scenes[other].read(p));
        }
        return { content: content ?? '' };
      },
    }),
  );

  logger.info('[memory] 工具已注册: memory_commit / memory_search / conversation_search / memory_read_scene');
}

/** 工具执行发生在 turn/end 前；确定性 L0 id 可提前引用当前轮真实 user 消息。 */
function currentTurnSourceIds(
  events: ReadonlyArray<{ type: string; seq: number; data?: unknown }>,
  sessionId: string,
): string[] {
  let start = -1;
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i].type === 'turn/start') { start = i; break; }
  }
  return events.slice(start + 1)
    .filter((e) => {
      if (e.type !== 'user/message') return false;
      const data = e.data as { source?: { kind?: string } } | undefined;
      return data?.source?.kind === 'user';
    })
    .map((e) => `l0:${sessionId}:${e.seq}`);
}

function renderMemoryItems(
  items: Array<{ id?: string; content?: string; type?: string; scene_name?: string; score?: number }>,
): string {
  if (!items || items.length === 0) return '（没有找到相关记忆）';
  return items
    .map((it, i) => `${i + 1}. [${it.type ?? ''}]${it.scene_name ? ` (${it.scene_name})` : ''}${it.id ? ` id=${it.id}` : ''} ${it.content ?? ''}`)
    .join('\n');
}

function renderConversationItems(
  items: Array<{ session_id?: string; role?: string; content?: string; timestamp?: number }>,
): string {
  if (!items || items.length === 0) return '（没有找到相关对话）';
  return items
    .map((it, i) => {
      const time = it.timestamp ? new Date(it.timestamp).toISOString() : '';
      return `${i + 1}. [${it.role ?? ''}]${time ? ` ${time}` : ''} (session=${it.session_id ?? ''})\n${it.content ?? ''}`;
    })
    .join('\n\n');
}

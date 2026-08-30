/**
 * 召回 Hook（消息侧注入版，ADR-0001）：
 * 1. agent/pre-step（waterfall，prepend 注册）：有新的用户来源消息到达的步骤
 *    （轮首 claim 或 steering 插话）→ 检索 L1 → 以合成消息形式注入到用户消息之前。
 *    注入消息携带插件来源（form: 'recall'），对用户可见、随会话历史持久累积——
 *    这就是"记忆生效"的显式提示（dsh-time-context 同款官方范式）。
 *    纯工具步透传；召回超时（recall.timeoutMs 总预算）跳过本轮，绝不阻塞对话。
 * 2. agent/created：在 agent 作用域注册动态上下文 provider（仅 L3 画像 + L2 场景导航）。
 *    记忆工具的选择与调用约束属于工具定义，不重复注入消息或动态系统上下文。
 *
 * 注入内容使用 <relevant-memories> 标签包裹（模型侧语义边界 + 助手回显剥离锚点），
 * 召回预算（单条/整轮）超限截断并引导模型用记忆工具查全文。
 * L0 防污染零成本：capture 侧只收 source.kind === 'user' 的消息，注入消息天然被排除。
 *
 * 注意：PromptContext.text 是**同步**函数，画像/场景导航这类异步文件读取必须走内存缓存，
 * 由定时刷新 + 管线更新后的主动失效来更新。
 */
import type { Context } from '@deepseek-ai/cordis';
import type {} from '@deepseek-ai/dsh-session'; // ctx.sessions 声明合并（回填读 live 会话）
import type { SessionId } from '@deepseek-ai/dsh-session';
import type { Agent } from '@deepseek-ai/dsh-agent';
import { createUserMessage } from '@deepseek-ai/dsh-llm';
import type { MemoryConfig } from '../config.js';
import type { LiveSettingsHandle } from '../settings.js';
import type { L1Store } from '../store/l1.js';
import type { ArchiveStore } from '../store/archive.js';
import type { PersonaStore } from '../store/persona.js';
import { RecallDedupeStore } from '../store/recall-dedupe.js';
import { OccupancyStore } from '../store/occupancy.js';
import type { SceneStore } from '../store/scenes.js';
import type { SessionModeStore } from '../store/session-modes.js';
import type { SessionLineageStore } from '../store/session-lineage.js';
import type { L1Hit, MemoryLogger } from '../types.js';
import { applyRecallBudget, raceRecallTimeout, RECALL_EMBED_CAP_MS } from '../util/recall-budget.js';
import {
  clearProfileShare,
  emptyOccupancyLedger,
  estimateInjectedMessageTokens,
  estimateStableSectionTokens,
  recordProfileShare,
  recordRecallInjection,
  resetForCompaction,
  type OccupancyLedger,
} from '../util/context-occupancy.js';
import { errDetail } from '../util/filelog.js';
import { blocksToText } from '../util/text.js';

const PROFILE_TTL = 60_000;

/* ── 召回回填的持久化服务兜底（票08）─────────────────────────────────────
 * 仅查看的旧会话不在 ctx.sessions live store（不发言就没有活 Session），surface
 * 路径拿不到。兜底走官方持久化服务 ctx.sessionPersistence.loadStored(id)——
 * "cwd 未知时跨项目目录读存储前缀"，多帧 zstd / packed 行 / torn 尾 / 项目槽
 * 全部由官方后端处理。窗口近似：存储前缀没有派生 surface，遇 compaction 类
 * 事件把此前累计清零（与账本 resetForCompaction 同语义，宁低勿高）。
 * 结果按会话进程内缓存（查看态会话不再增长；一旦发言则由 live 路径接管）。
 */
interface StoredLike {
  events?: ReadonlyArray<{
    type?: string;
    data?: { source?: Record<string, unknown>; content?: ReadonlyArray<{ type?: string; text?: string }> };
  }>;
}
const storedEstimateCache = new Map<string, number | null>();



/** 召回查询只取会话末尾 N 条消息（长会话每步把全史拼进 FTS MATCH 会让检索成本线性上涨）。 */
const RECALL_QUERY_TAIL_MESSAGES = 8;
/** 召回查询总字符上限（保留末尾——最新语境权重最高）。 */
const RECALL_QUERY_MAX_CHARS = 2_000;
/** 自动注入宁缺毋滥；更多结果由模型显式 memory_search 获取。 */
export const AUTO_RECALL_MAX_RESULTS = 2;
/** 成功注入后的自动召回冷却：时间与轮次两个条件都满足才重新放行。 */
export const AUTO_RECALL_COOLDOWN_MS = 10 * 60_000;
export const AUTO_RECALL_COOLDOWN_TURNS = 3;
const AUTO_RECALL_COOLDOWN_SESSION_CAP = 200;

export interface RecallCooldownState {
  lastInjectedAt: number;
  turnsSinceInjection: number;
}

/** 每个新用户步推进一次；无既往注入立即放行，否则时间与轮次缺一不可。 */
export function advanceRecallCooldown(
  state: RecallCooldownState | undefined,
  now: number,
  minIntervalMs = AUTO_RECALL_COOLDOWN_MS,
  minTurns = AUTO_RECALL_COOLDOWN_TURNS,
): boolean {
  if (!state) return true;
  state.turnsSinceInjection++;
  return now - state.lastInjectedAt >= minIntervalMs && state.turnsSinceInjection >= minTurns;
}

/** registerRecall 的时钟/阈值测试缝；生产调用不传，固定使用上面的保守默认值。 */
export interface RecallCadenceOptions {
  now?: () => number;
  minIntervalMs?: number;
  minTurns?: number;
}

/**
 * 从会话消息构建召回查询（纯函数）：末尾 N 条 + 总长截断，空输入返回空串。
 * 全史拼接会让 MATCH 表达式随会话长度线性膨胀（整会话累计二次方成本）。
 */
export function buildRecallQuery(
  messages: Array<{ content: unknown }>,
  tailMessages = RECALL_QUERY_TAIL_MESSAGES,
  maxChars = RECALL_QUERY_MAX_CHARS,
): string {
  const tail = messages.slice(-tailMessages);
  let text = tail.map((m) => blocksToText(m.content as never)).join(' ').trim();
  if (text.length > maxChars) text = text.slice(-maxChars);
  return text;
}

/**
 * 低信息输入不值得触发跨会话检索。这里只拦极短输入与纯确认/推进语，
 * 不尝试做主题分类；真正的相关性仍由 L1 strict 检索门槛判断。
 */
export function isLowInformationRecallQuery(query: string): boolean {
  const compact = query.normalize('NFKC').toLowerCase().replace(/[\s\p{P}\p{S}]+/gu, '');
  if ([...compact.matchAll(/[\p{L}\p{N}]/gu)].length < 3) return true;
  if (/^(?:好的?|好吧|行吧?|可以|没问题|开始吧|继续|continue|嗯+|哦+|呜+|哈+|谢谢|收到|算了|就这样吧?|知道了|明白了|对|是的|不是|再试试|试试|看看)$/.test(compact)) return true;
  return /^(?:好的?|好吧|行吧?|可以|没问题|收到|知道了|明白了)(?:开始吧|继续|continue|再试试|试试|看看)$/.test(compact);
}

/** 单会话召回统计（悬浮卡信息区数据源；每轮 O(1) 记账，agent/disposed 清理）。
 *  口径声明：这是"注入统计"而非 bench 的离线 recall@k——运行时没有 ground truth，
 *  命中率 = hitTurns / injectedTurns。去重语义（0.8.6）：全量压制轮计入 hitTurns
 *  （相关记忆已在模型上下文里，本质是命中而非未命中），injectedTurns 仍计全部
 *  发生过检索的轮次（保住分母语义与悬浮卡口径连续性）。 */
// RecallSessionStats 已迁入契约单一事实源（src/contract.ts）——host 悬浮卡端点与
// client 悬浮卡共享同一形状；import type 供本地使用，re-export 不断裂既有引用。
import type { RecallSessionStats } from '../contract.js';
export type { RecallSessionStats } from '../contract.js';

/** 新建零值统计（首次出现的会话）。 */
export function emptyRecallStats(now = Date.now()): RecallSessionStats {
  return {
    injectedTurns: 0,
    hitTurns: 0,
    totalHits: 0,
    timeouts: 0,
    suppressedRecalls: 0,
    lastHits: 0,
    lastDurationMs: 0,
    updatedAt: now,
  };
}

export interface RecallHooks {
  /** 管线更新后调用：画像/场景缓存立即失效并异步刷新。 */
  invalidateProfile(): void;
  /** 会话召回统计只读视图（未发生过检索的会话返回 undefined）。 */
  stats(sessionId: string): RecallSessionStats | undefined;
  /** 记忆占用账本只读出口：内存优先，miss 时从流水复生（重启后历史会话）；从未注入返回 null。 */
  occupancy(sessionId: string): OccupancyLedger | null;
  /**
   * 稳定区份额估算（票08 旧会话回填）：按当前画像/导航/指南组词折算 token，
   * 纯读不记账——旧会话系统提示里"现在大约坐着多少稳定区"的最佳可得估计。
   */
  estimateProfileTokens(agentId: string): number;
  /** 召回份额回填（票08）：live 会话 surface 现扫本插件注入，miss 时读盘上日志兜底；
   *  均不可得返回 null。 */
  estimateRecallTokens(sessionId: string): Promise<number | null>;
}

export function registerRecall(
  ctx: Context,
  cfg: MemoryConfig,
  stores: { archive?: ArchiveStore; l1: L1Store; scenes: Record<'chat' | 'work', SceneStore>; persona: Record<'chat' | 'work', PersonaStore> },
  logger: MemoryLogger,
  live: LiveSettingsHandle,
  modes: SessionModeStore,
  dataDir: string,
  lineage?: SessionLineageStore,
  cadence: RecallCadenceOptions = {},
): RecallHooks {
  /** 召回去重存储（同会话已注入的记忆不再重复注入；写穿持久化，重启不丢）。 */
  const dedupe = new RecallDedupeStore(dataDir, logger);
  /** 记忆占用流水（账本迁移写穿；重启后历史会话账目由此复生——票07）。 */
  const occupancyStore = new OccupancyStore(dataDir, logger);
  /** 每 agent 召回统计（悬浮卡信息区与诊断使用）。 */
  const recallStats = new Map<string, RecallSessionStats>();
  /** 成功注入后的进程内冷却；保留 agent dispose/恢复语义，LRU 上限防会话常驻增长。 */
  const cooldowns = new Map<string, RecallCooldownState>();
  const cadenceNow = cadence.now ?? Date.now;
  const cooldownMs = cadence.minIntervalMs ?? AUTO_RECALL_COOLDOWN_MS;
  const cooldownTurns = cadence.minTurns ?? AUTO_RECALL_COOLDOWN_TURNS;
  const markCooldown = (id: string): void => {
    cooldowns.delete(id);
    cooldowns.set(id, { lastInjectedAt: cadenceNow(), turnsSinceInjection: 0 });
    while (cooldowns.size > AUTO_RECALL_COOLDOWN_SESSION_CAP) {
      const oldest = cooldowns.keys().next().value as string | undefined;
      if (!oldest) break;
      cooldowns.delete(oldest);
    }
  };
  const statFor = (id: string): RecallSessionStats => {
    let s = recallStats.get(id);
    if (!s) {
      s = emptyRecallStats();
      recallStats.set(id, s);
    }
    return s;
  };
  /** 每 agent 记忆占用账本（权威账本的唯一宿主实例；占用指示器与悬浮卡同源消费）。 */
  const occupancyByAgent = new Map<string, OccupancyLedger>();
  const ledgerFor = (id: string): OccupancyLedger => {
    let led = occupancyByAgent.get(id);
    if (!led) {
      // 进程重启/agent 重建后回看：从流水复生（新迁移在持久值上继续累加——票07）
      led = occupancyStore.load(id) ?? emptyOccupancyLedger();
      occupancyByAgent.set(id, led);
    }
    return led;
  };
  // 画像/场景导航按族缓存（分族隔离：注入时按会话档位选族）
  const profileCache: Record<'chat' | 'work', ProfileParts> = {
    chat: { persona: '', nav: '' },
    work: { persona: '', nav: '' },
  };

  const refreshProfile = async (): Promise<void> => {
    try {
      const [chat, work] = await Promise.all([
        loadProfileParts(stores, cfg, 'chat'),
        loadProfileParts(stores, cfg, 'work'),
      ]);
      profileCache.chat = chat;
      profileCache.work = work;
    } catch (err) {
      logger.warn(`[memory] 画像/场景缓存刷新失败: ${errDetail(err)}`);
    }
  };

  // 初始刷新 + 定时刷新（TTL）
  void refreshProfile();
  ctx.effect(() => {
    const timer = setInterval(() => void refreshProfile(), PROFILE_TTL);
    return () => clearInterval(timer);
  });

  const invalidateProfile = (): void => {
    void refreshProfile();
  };

  // agent 销毁时清掉召回统计槽（去重记录不随 agent 清——持久化语义：会话恢复后继续压制）
  ctx.on('agent/disposed', (payload) => {
    recallStats.delete(payload.agent.id);
    occupancyByAgent.delete(payload.agent.id);
  });

  // 上下文压缩/清空 → 已注入内容从模型上下文丢失，重置该会话的去重压制
  // （resume/startup 不重置：历史仍在，已注入的记忆模型还持有）。
  // 占用账本同步全量归零（宁低勿高；v1 轮级粒度近似，事件级 shadow price 对齐留待后续）。
  ctx.on('agent/session-start', (payload) => {
    if (payload.source === 'compact' || payload.source === 'clear') {
      dedupe.reset(payload.agent.id);
      cooldowns.delete(payload.agent.id);
      const led = ledgerFor(payload.agent.id);
      resetForCompaction(led);
      occupancyStore.save(payload.agent.id, led); // stock 归零 ⇒ 流水条目删除
      logger.info(`[memory] 召回去重与占用账本重置（agent=${payload.agent.id}，source=${payload.source}）`);
    }
  });

  // ── 1. pre-step 消息侧注入：记忆先行于每一条新的用户输入（ADR-0001） ──
  // prepend 注册 + 先 next() 再改写：不劫持其他监听器（dsh-time-context 官方范式）。
  if (cfg.recall.enabled) {
    ctx.on(
      'agent/pre-step',
      async (payload, next) => {
        const decision = await next();
        if (decision.kind === 'reject' || payload.signal.aborted) return decision;
        try {
          const s = live.get();
          const mode = modes.get(payload.agent.id);
          // 三级读闸：主闸 → off 档（完全隐身）→ 注入开关（#38：会话覆盖 ?? 全局）
          if (!s.enabled || mode === 'off' || !modes.resolvedRecall(payload.agent.id, s.recall)) return decision;
          // 只在有新的用户来源消息的步骤注入（轮首 claim 或 steering 插话）；纯工具步透传
          const hasNewUserMessage = decision.messages.some(
            (m) => (m as { source?: { kind?: string } }).source?.kind === 'user',
          );
          if (!hasNewUserMessage) return decision;
          const cooldown = cooldowns.get(payload.agent.id);
          if (!advanceRecallCooldown(cooldown, cadenceNow(), cooldownMs, cooldownTurns)) {
            const skipped = statFor(payload.agent.id);
            skipped.lastHits = 0;
            skipped.updatedAt = cadenceNow();
            logger.debug?.(
              `[memory] 自动召回冷却中（agent=${payload.agent.id}，` +
              `${cooldown?.turnsSinceInjection ?? 0}/${cooldownTurns} 轮，` +
              `${Math.max(0, cadenceNow() - (cooldown?.lastInjectedAt ?? 0))}/${cooldownMs}ms）`,
            );
            return decision;
          }
          // 只拿本步新用户消息做查询：历史回复与旧 recall 已在当前上下文里，重复拼入会
          // 造成检索自反馈，并让一个旧主题在后续每轮持续触发注入。
          const newUserMessages = decision.messages.filter(
            (m) => (m as { source?: { kind?: string } }).source?.kind === 'user',
          );
          const query = buildRecallQuery(newUserMessages as Array<{ content: unknown }>);
          // 空查询是退化轮（无用户文本），重置命中信号但不计入统计
          if (!query || isLowInformationRecallQuery(query)) {
            const degenerate = statFor(payload.agent.id);
            degenerate.lastHits = 0;
            return decision;
          }
          const st = statFor(payload.agent.id);
          st.injectedTurns++;
          st.lastHits = 0;
          st.updatedAt = Date.now();
          const searchStart = Date.now();
          const visibleSessionIds = lineage?.ancestors(payload.agent.id) ?? [payload.agent.id];
          const resultLimit = Math.min(cfg.recall.maxResults, AUTO_RECALL_MAX_RESULTS);
          const hits = await raceRecallTimeout(
            Promise.all([
              stores.l1.search(query, resultLimit, {
                scoreThreshold: cfg.recall.scoreThreshold,
                strictThreshold: true,
                family: mode === 'auto' ? undefined : mode,
                // 嵌入内层钳制：给 FTS 降级留出总预算内的时间（远程限 HTTP fetch；本地经 worker 代理 race 放弃）
                embeddingTimeoutMs: RECALL_EMBED_CAP_MS,
                visibleSessionIds,
                visiblePresetId: lineage?.presetOf(payload.agent.id),
              }),
              Promise.resolve(cfg.archive?.autoRecall && stores.archive
                ? stores.archive.search(query, 1, visibleSessionIds, true)
                : []),
            ]).then(([memoryHits, archiveHits]) => {
              const archives = archiveHits.map((hit) => ({
                id: hit.id,
                content: `${hit.summary}\n（档案 ${hit.id}；需要原话时调用 conversation_search 并传 archive_id）`,
                type: 'archive',
                scene_name: `${new Date(hit.bucketStart).toISOString()}..${new Date(hit.latestAt).toISOString()}`,
                score: hit.score,
              }));
              // 档案命中严格过滤后优先占一个槽；余量留给稳定 L1，避免两类互相挤没。
              return [...archives, ...memoryHits].slice(0, resultLimit);
            }),
            cfg.recall.timeoutMs,
          );
          st.updatedAt = Date.now();
          if (hits === undefined) {
            st.timeouts++;
            logger.warn('[memory] 召回超时，跳过本轮注入（不阻塞对话）');
            return decision;
          }
          st.lastDurationMs = Date.now() - searchStart;
          // 召回去重：同会话已注入过的记录不再重复注入（模型上下文已持有，省 token）。
          // 纯过滤——剩几条注几条，全量压制（0 条新鲜命中）是正确状态而非未命中。
          const seen = dedupe.seen(payload.agent.id);
          const fresh = hits.filter((h) => !seen.has(h.id)).slice(0, AUTO_RECALL_MAX_RESULTS);
          const suppressed = hits.length - fresh.length;
          st.suppressedRecalls += suppressed;
          if (suppressed > 0) {
            logger.debug?.(
              `[memory] 召回去重：压制 ${suppressed} 条已注入记忆（agent=${payload.agent.id}，余 ${fresh.length} 条新鲜命中）`,
            );
          }
          if (hits.length > 0) {
            // 全量压制轮也计入命中：相关记忆已在模型上下文里，本质是命中
            st.hitTurns++;
            st.totalHits += fresh.length;
          }
          if (fresh.length === 0) return decision;
          const lines = applyRecallBudget(
            fresh.map((h) => `- [${h.scene_name ? `${h.type}|${h.scene_name}` : h.type}] ${h.content}`),
            { maxCharsPerMemory: cfg.recall.maxCharsPerMemory, maxTotalRecallChars: cfg.recall.maxTotalRecallChars },
          );
          if (lines.length === 0) return decision;
          // 预算截断只丢尾部（前缀保留）：实际注入 = fresh 的前 lines.length 条——只标记模型真实看到的
          dedupe.mark(payload.agent.id, fresh.slice(0, lines.length).map((h) => h.id));
          st.lastHits = lines.length;
          const text = [
            '<relevant-memories>',
            '以下是当前对话召回的相关记忆，不代表当前任务进程，仅作为参考：',
            '',
            ...lines,
            '',
            '</relevant-memories>',
          ].join('\n');
          logger.info(
            `[memory] 召回注入 ${lines.length} 条记忆/档案（mode=${mode}，query="${query.slice(0, 30).replace(/\n/g, ' ')}…"，agent=${payload.agent.id}，消息侧）`,
          );
          const injection = createUserMessage({
            content: [{ type: 'text', text }],
            // plugin 字段是宿主 UI 的署名后缀（"上下文注入 · memory"）——用展示友好的
            // 子系统名，不用 cordis id（dsh-memory）；kind:'plugin' 的标题恒为通用
            // "上下文注入"（专用"跨会话召回"标题仅留给 session-reference 来源）
            source: { kind: 'plugin', plugin: 'memory', form: 'recall' },
          });
          // 入账在成功构造注入消息之后、返回 enter 之前——任何前置抛错路径账目零扰动
          const led = ledgerFor(payload.agent.id);
          recordRecallInjection(led, text.length);
          occupancyStore.save(payload.agent.id, led);
          markCooldown(payload.agent.id);
          // 注入消息排在用户新消息之前（原版 prepend 语义：先线索后问题）
          return { kind: 'enter', messages: [injection, ...decision.messages] };
        } catch (err) {
          logger.warn(`[memory] 召回注入失败（跳过本轮）: ${errDetail(err)}`);
          return decision;
        }
      },
      { prepend: true },
    );
  }

  // ── 2. agent 作用域上下文 provider（系统提示稳定区：画像 + 导航 + 门控指南） ──
  // 插件可能在默认 agent 创建之后才加载（组合顺序由依赖决定），
  // 因此除了监听 agent/created，还要给已存在的 agent 补注册。
  /**
   * 稳定区当前组词（纯读，不记账）：text() 的取词部分单独成函数，
   * 供旧会话回填估算（RecallHooks.estimateProfileTokens）复用同一口径。
   */  const composeStableText = (agentId: string): string => {
    const s = live.get();
    const mode = modes.get(agentId);
    // 与 pre-step 同款三级读闸（#38）：主闸 → off 档 → 注入开关；空串即物理离场
    if (!s.enabled || mode === 'off' || !modes.resolvedRecall(agentId, s.recall)) return '';
    // auto 档：两族按类别归组（画像/导航各一个标签，域内 <domain> 分块）；纯档：单族原格式
    const body =
      mode === 'auto'
        ? formatProfileAuto(profileCache.chat, profileCache.work)
        : formatProfileSingle(profileCache[mode]);
    return body;
  };

  async function estimateRecallFromStorage(sessionId: string): Promise<number | null> {
    if (storedEstimateCache.has(sessionId)) return storedEstimateCache.get(sessionId) ?? null;
    let tokens: number | null = null;
    try {
      // 可选服务（JSONL 后端注册名）；缺失/其它实现 → 回填隐藏
      const persistence = (await (ctx as unknown as { get?: (name: string) => unknown }).get?.('sessionPersistence')) as
        | { loadStored?: (id: never, signal?: AbortSignal) => Promise<StoredLike | undefined> }
        | undefined;
      const stored = typeof persistence?.loadStored === 'function' ? await persistence.loadStored(sessionId as never) : undefined;
      if (stored?.events) {
        tokens = 0;
        for (const ev of stored.events) {
          if (typeof ev.type === 'string' && ev.type.startsWith('compaction')) tokens = 0;
          if (ev.type !== 'user/message') continue;
          const src = ev.data?.source;
          if (!src || src.kind !== 'plugin' || src.plugin !== 'memory' || src.form !== 'recall') continue;
          let chars = 0;
          for (const b of ev.data?.content ?? []) {
            if (b?.type === 'text' && typeof b.text === 'string') chars += b.text.length;
          }
          if (chars > 0) tokens += estimateInjectedMessageTokens(chars);
        }
      }
    } catch {
      tokens = null;
    }
    storedEstimateCache.set(sessionId, tokens);
    return tokens;
  }

  /**
   * 召回份额回填（票08 旧会话）：live 会话的 surface（模型可见序号集）∩ 全事件日志
   * 里本插件的 recall 注入，官方同式折算。窗口语义天然正确——被压缩折叠的注入不在
   * surface.nodes 上，自动出局。会话不在 live store（未打开）返回 null。
   */
  const estimateRecallTokens = async (sessionId: string): Promise<number | null> => {
    try {
      // cordis 属性访问（ctx.sessions）对未 inject 的服务抛 "without inject"（实测）；
      // 可选服务一律走 ctx.get() 的宽容路径
      const sessions = ctx.get?.('sessions') as
        | { get?: (id: SessionId) => { surface: { nodes: readonly number[] }; events: ReadonlyArray<{ type: string; seq: number; data?: { source?: Record<string, unknown>; content?: ReadonlyArray<{ type?: string; text?: string }> } }> } | undefined }
        | undefined;
      const session = typeof sessions?.get === 'function' ? sessions.get(sessionId as SessionId) : undefined;
      if (session) {
        const visible = new Set(session.surface.nodes);
        let total = 0;
        for (const ev of session.events) {
          if (ev.type !== 'user/message' || !visible.has(ev.seq)) continue;
          const msg = ev.data as
            | { source?: Record<string, unknown>; content?: ReadonlyArray<{ type?: string; text?: string }> }
            | undefined;
          const src = msg?.source;
          if (!src || src.kind !== 'plugin' || src.plugin !== 'memory' || src.form !== 'recall') continue;
          let chars = 0;
          for (const b of msg.content ?? []) {
            if (b?.type === 'text' && typeof b.text === 'string') chars += b.text.length;
          }
          if (chars > 0) total += estimateInjectedMessageTokens(chars);
        }
        return total;
      }
      // 仅查看的旧会话不在 live store：官方持久化服务读存储前缀兜底（见函数头）
      return estimateRecallFromStorage(sessionId);
    } catch {
      return null; // 服务缺失/形状异常：回填隐藏，不扰动主流程
    }
  };

  const registered = new WeakSet<object>();
  // context() 的 disposer 必须挂到插件自身生命周期：agent.ctx 比插件实例活得久，
  // 不主动清理会导致热重载后旧注册泄漏、新实例撞名（"already registered"）
  const contextDisposers: Array<() => void> = [];
  const registerForAgent = (agent: Agent): void => {
    if (registered.has(agent)) return;
    registered.add(agent);
    try {
      contextDisposers.push(
        agent.ctx.systemPrompt.context({
          name: 'memory:profile',
          order: 510,
          text: () => {
            const final = composeStableText(agent.id);
            const ledger = ledgerFor(agent.id);
            // 空串即物理离场（停用/OFF/门控全空）：份额同边界清零；否则按实际长度入账
            if (final === '') clearProfileShare(ledger);
            else recordProfileShare(ledger, final.length);
            occupancyStore.save(agent.id, ledger);
            return final;
          },
        }),
      );
    } catch (err) {
      logger.warn(`[memory] 召回上下文注册失败: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const agents = ctx.get('agents');
  if (agents) {
    for (const agent of agents.list()) registerForAgent(agent);
  }
  ctx.on('agent/created', (payload) => {
    registerForAgent(payload.agent);
  });
  ctx.effect(() => () => {
    for (const dispose of contextDisposers.splice(0)) {
      try {
        dispose();
      } catch {
        /* agent 可能已先一步销毁 */
      }
    }
  });

  return {
    invalidateProfile,
    stats: (id) => recallStats.get(id),
    /** 占用账本只读出口：内存优先，miss 时从流水复生（重启后历史会话）；从未注入返回 null。 */
    occupancy: (id): OccupancyLedger | null => {
      const led = occupancyByAgent.get(id) ?? occupancyStore.load(id);
      if (led) occupancyByAgent.set(id, led);
      return led ?? null;
    },
    estimateProfileTokens: (id: string): number =>
      estimateStableSectionTokens(composeStableText(id).length),
    estimateRecallTokens,
  };
}

/** 单族画像/导航片段（注入侧按档位组装，纯档与 auto 档格式不同）。 */
interface ProfileParts {
  persona: string;
  nav: string;
}

/** auto 档 <user-persona> 内的域说明：让模型理解分块结构与两域的独立性。 */
const DOMAIN_HINT =
  '以下内容按记忆域分块：chat=用户个人画像（User Narrative Profile），work=团队工作准则（Team Operating Doctrine）。' +
  '两域独立蒸馏与更新，请按当前对话语境参考对应域，不要把一域的内容当作另一域的事实。';

function wrapDomain(family: 'chat' | 'work', content: string): string {
  const label = family === 'chat' ? '用户个人画像' : '团队工作准则';
  return `<domain family="${family}" label="${label}">\n${content.trim()}\n</domain>`;
}

/** 纯档注入：单族画像 + 场景导航（沿用原有格式）。 */
function formatProfileSingle(parts: ProfileParts): string {
  const segments: string[] = [];
  if (parts.persona) segments.push(`<user-persona>\n${parts.persona}\n</user-persona>`);
  if (parts.nav) segments.push(`<scene-navigation>\n${parts.nav}\n</scene-navigation>`);
  return segments.join('\n\n');
}

/** auto 档注入：两族按类别归组——画像共用一个 <user-persona>、导航共用一个 <scene-navigation>，域内 <domain> 分块。 */
function formatProfileAuto(chat: ProfileParts, work: ProfileParts): string {
  const segments: string[] = [];
  const personas: Array<['chat' | 'work', string]> = [
    ['chat', chat.persona],
    ['work', work.persona],
  ];
  const personaBlocks = personas.filter(([, p]) => p.trim()).map(([f, p]) => wrapDomain(f, p));
  if (personaBlocks.length > 0) {
    segments.push(`<user-persona>\n${DOMAIN_HINT}\n\n${personaBlocks.join('\n\n')}\n</user-persona>`);
  }
  const navs: Array<['chat' | 'work', string]> = [
    ['chat', chat.nav],
    ['work', work.nav],
  ];
  const navBlocks = navs.filter(([, n]) => n.trim()).map(([f, n]) => wrapDomain(f, n));
  if (navBlocks.length > 0) {
    segments.push(`<scene-navigation>\n${navBlocks.join('\n\n')}\n</scene-navigation>`);
  }
  // 注意：工具指南由注入侧统一附加一次（auto 档不重复）
  return segments.join('\n\n');
}

async function loadProfileParts(
  stores: { scenes: Record<'chat' | 'work', SceneStore>; persona: Record<'chat' | 'work', PersonaStore> },
  cfg: MemoryConfig,
  family: 'chat' | 'work',
): Promise<ProfileParts> {
  const persona = cfg.recall.includePersona ? ((await stores.persona[family].read()) ?? '') : '';
  const nav = cfg.recall.includeSceneNav ? ((await stores.scenes[family].navigation()) ?? '').trim() : '';
  return { persona, nav };
}
